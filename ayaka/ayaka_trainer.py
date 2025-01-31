import jax, flax, math, torch, optax
from jax import numpy as jnp, random as jrand, Array
from flax import nnx
from jax.sharding import NamedSharding as NS, Mesh, PartitionSpec as PS
from jax.experimental import mesh_utils

import numpy as np
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm.auto import tqdm
from functools import partial
from flax.training import train_state
import flax.traverse_util, builtins
from flax.serialization import to_state_dict, from_state_dict
from flax.core import freeze, unfreeze

import json, os, wandb, gc, time, pickle, click
import matplotlib.pyplot as plt
from PIL import Image
from safetensors import safe_open
from cosmos_tokenizer.image_lib import ImageTokenizer
from huggingface_hub import login, snapshot_download, file_download


from ar_model import ART, rms_norm, config as mcfg, token_match_accuracy


# XLA/JAX flags
jax.config.update("jax_default_matmul_precision", "bfloat16")
JAX_TRACEBACK_FILTERING = "off"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
XLA_PYTHON_CLIENT_MEM_FRACTION = 0.20
JAX_DEFAULT_MATMUL_PRECISION = "bfloat16"


class config:
    batch_size = 128
    img_size = 32
    seed = 333
    data_split = 1_000_000
    cfg_scale = 4.0


# keys/env seeds
rkey = jrand.key(config.seed)
randkey = jrand.key(config.seed)
rngs = nnx.Rngs(config.seed)


# mesh / sharding configs
num_devices = jax.device_count()
devices = jax.devices()
num_processes = jax.process_count()
rank = jax.process_index()


print(f"found {num_devices} JAX device(s)")
for device in devices:
    print(f"{device} / ")


mesh_devices = mesh_utils.create_device_mesh((num_devices,))
mesh = Mesh(mesh_devices, axis_names=("data"))
data_sharding = NS(mesh, PS("data"))
rep_sharding = NS(mesh, PS())

model_name = "Cosmos-0.1-Tokenizer-DI8x8"
hf_repo = f"nvidia/{model_name}"
local_dir = "checks"
# os.makedirs(local_dir, exist_ok=True)
decoder = ImageTokenizer(checkpoint_dec=f"{local_dir}/cosmos/decoder.jit", device="cpu")
print("loaded tokenizer")



class ImageTokenDataset(IterableDataset):

    def __init__(
        self,
        safetensor_path="./data/cosmos_imagenet/imagenet_di8x8.safetensors",
        debug=True,
    ):
        self.safetensor_path = safetensor_path
        self.split = 100

        metadata_path = safetensor_path.replace(".safetensors", "_metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
            self.total_samples = self.metadata["total_samples"]

        if debug:
            self.total_samples = 16

        with safe_open(safetensor_path, framework="flax") as f:
            self.indices = (
                f.get_tensor("indices")
                .astype(jnp.uint16)[: self.split]
                .astype(jnp.int64)
            )
            self.labels = f.get_tensor("labels")[: self.split].astype(jnp.int32)

        if debug:
            self.indices = self.indices[: self.total_samples]
            self.labels = self.labels[: self.total_samples]

    def __len__(self):
        return int(self.total_samples)

    def __iter__(self):
        for tokens, labels in zip(self.indices, self.labels):
            indices = tokens.reshape(-1)
            class_label = labels

            yield {"input_ids": indices, "class_label": class_label}


def decode(data):
    # print(f'decoder input shape = {data.shape}')
    data = torch.from_numpy(np.array(data))
    data = data.reshape(1, 32, 32)  # .long()
    # Decode the image
    with torch.no_grad():
        reconstructed = decoder.decode(data)
        # print(f'{reconstructed.shape = }')

    img = ((reconstructed[0].cpu().float() + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    img = img.permute(1, 2, 0).numpy()
    img = Image.fromarray(img)

    return img


def inspect_latents(batch):
    batch = [decode(x) for x in batch]
    file = f"images/imagenet-cosmos-16.jpg"
    gridfile = image_grid(batch, file)
    print(f"sample saved @ {gridfile}")


def image_grid(pil_images, file, grid_size=(4, 4), figsize=(4, 4)):
    rows, cols = grid_size
    assert len(pil_images) <= rows * cols, "Grid size must accommodate all images."

    # Create a matplotlib figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten for easy indexing

    for i, ax in enumerate(axes):
        if i < len(pil_images):
            # Convert PIL image to NumPy array and plot
            ax.imshow(np.array(pil_images[i]))
            ax.axis("off")  # Turn off axis labels
        else:
            ax.axis("off")  # Hide empty subplots for unused grid spaces

    plt.tight_layout()
    plt.savefig(file, bbox_inches="tight")
    plt.show()

    return file

keyrng = jrand.PRNGKey(config.seed)

@partial(
    nnx.jit,
    in_shardings=(rep_sharding, data_sharding),
    # out_shardings=(None,),
)
def generate(self, class_labels: Array):
    max_tokens = 1024
    temp = 1.0
    cfg_scale = 4.0
    topk = None

    b = class_labels.shape[0]

    class_labels = jnp.tile(class_labels, (2,))
    class_labels = class_labels.at[b].set(1000)
    x = (class_labels + 65536)[:, None]

    kv_caches = [None] * len(self.layers)
    x_init = self.token_embed(x)
    freq = self.rotary(x_init, hw=(32, 32))
    cos, sin = freq
    cos, sin = cos.value, sin.value

    x_all = jnp.zeros((b * 2, max_tokens + 1)).astype(
        jnp.int64
    )  # jnp.broadcast_to(x, (2*b, max_tokens + 1))

    init_carry = {
        "x_all": x_all,  # Now x_all is pre-allocated
        "kv_caches": kv_caches,
        "current_index": 1,  # Start filling from the second column (index 1)
        "randkey": jrand.key(config.seed),
        "x_init_emb": x_init,  # Pass initial embedding for addition
    }

    def loop_fn(i, carry):
        # print(i)
        x_all = carry["x_all"]
        # print(f'{x_all.shape = }')
        kv_caches = carry["kv_caches"]
        current_index = carry["current_index"]
        randkey = carry["randkey"]
        x_init_emb = carry["x_init_emb"]
        b = x_all.shape[0] // 2  # Get batch size

        x_emb = self.token_embed(x_all[:, -1:])
        x_emb = x_emb + x_init

        x_emb = rms_norm(x_emb)
        cos_local = jax.lax.dynamic_slice_in_dim(cos, i, 1, axis=1)
        sin_local = jax.lax.dynamic_slice_in_dim(sin, i, 1, axis=1)

        freq_local = (cos_local, sin_local)
        v1 = None

        for n, layer in enumerate(self.layers):
            (x_emb, v1), new_kv_cache = layer(
                x_emb, kv_cache=kv_caches[n], freq=freq_local, v1=None, pos=i
            )
            kv_caches[n] = new_kv_cache

        x_emb = rms_norm(x_emb)
        logits = self.linear_head(x_emb).squeeze(1)
        # print(f'{logits.shape = }')

        logits_cond = logits[0::2, :]
        logits_uncond = logits[1::2, :]
        logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
        logits = logits / temp

        if topk is not None:
            v, _ = jax.lax.top_k(logits, min(topk, logits.shape[-1]))
            logits = jnp.where(logits < v[:, [-1]], -jnp.inf, logits)

        probs = nnx.softmax(logits, axis=-1)
        randkey, idx_key = jrand.split(randkey)  # Split random key
        idx_next = jrand.categorical(idx_key, probs, axis=1)
        print(f"{idx_next.shape = } / {x_all.shape = } / {probs.shape = }")

        idx_next = jnp.tile(idx_next, (2,))[:, None]
        
        # Update x_all at the current column index 'current_index'
        x_all = x_all.at[jnp.arange(x_all.shape[0]), current_index].set(idx_next[:, 0])

        new_carry = {
            "x_all": x_all,
            "kv_caches": kv_caches,
            "current_index": current_index + 1,  # Increment column index
            "randkey": randkey,
            "x_init_emb": x_init_emb,
        }
        return new_carry

    final_carry = jax.lax.fori_loop(0, max_tokens, loop_fn, init_carry)
    x_out = final_carry["x_all"][:, 1:]

    return x_out


def ar_sample_batch(step, model, batch):
    labels = batch["class_label"][:16]
    realtokens = batch["input_ids"][:16]

    print(f"sampling from labels {labels}")
    labels = jax.device_put(labels, data_sharding)

    image_batch = generate(model, labels)[:16]
    
    print("gen complete..decoding")

    file = f"arsamples/ayaka_{step}.png"
    batch = [decode(x) for x in image_batch]
    gridfile = image_grid(batch, file)
    print(f"saved @ {gridfile}")

    sample_token_match = token_match_accuracy(image_batch, realtokens).item() * 100
    print(f"{sample_token_match = }")
    jax.clear_caches()

    return gridfile, sample_token_match


def wandb_logger(key: str, project_name="ayaka_ar", run_name=None):  # wandb logger
    # initilaize wandb
    wandb.login(key=key)
    wandb.init(
        project=project_name,
        name=run_name or None,
        settings=wandb.Settings(init_timeout=120),
    )


def device_get_model(model):
    state = nnx.state(model)
    state = jax.device_get(state)
    nnx.update(model, state)

    return model


# save model params in pickle file
def save_paramdict_pickle(model, filename="model.ckpt"):
    params = nnx.state(model)
    params = jax.device_get(params)

    state_dict = to_state_dict(params)
    frozen_state_dict = freeze(state_dict)

    flat_state_dict = flax.traverse_util.flatten_dict(frozen_state_dict, sep=".")

    with open(filename, "wb") as f:
        pickle.dump(frozen_state_dict, f)

    return flat_state_dict


def load_paramdict_pickle(model, filename="model.ckpt"):
    with open(filename, "rb") as modelfile:
        params = pickle.load(modelfile)

    params = unfreeze(params)
    params = flax.traverse_util.unflatten_dict(params, sep=".")
    params = from_state_dict(model, params)
    nnx.update(model, params)

    return model, params


@partial(
    nnx.jit,
    in_shardings=(rep_sharding, rep_sharding, data_sharding, data_sharding),
    out_shardings=(None, None, None),
)
def train_step(model, optimizer, tokens, label):

    def loss_func(model, tokens, label):
        logits, loss, token_match = model(tokens, label)

        return loss, token_match

    gradfn = nnx.value_and_grad(loss_func, has_aux=True)
    (loss, token_match), grads = gradfn(model, tokens, label)
    grad_norm = optax.global_norm(grads)
    optimizer.update(grads)

    return loss, token_match * 100, grad_norm


def overfit(epochs, model, optimizer, train_loader, schedule):
    train_loss = 0.0
    model.train()

    batch = next(iter(train_loader))
    gridfile = ar_sample_batch("test", model, batch)

    wandb_logger(key="")
    stime = time.time()

    print("start overfitting.../")
    for epoch in tqdm(range(epochs)):
        lr = schedule(epoch)
        latent, text_embed = batch["input_ids"], batch["class_label"]
        train_loss, token_match, grad_norm = train_step(
            model, optimizer, latent, text_embed
        )

        print(
            f"step {epoch}, loss-> {train_loss.item():.5f}, grad_norm {grad_norm.item()}"
        )

        wandb.log(
            {
                "loss": train_loss.item(),
                "log_loss": math.log10(train_loss.item() + 1e-8),
                "token_match/train": token_match.item(),
                "grad_norm": grad_norm.item(),
                "log_grad_norm": math.log10(grad_norm.item() + 1e-8),
                "lr": lr,
            }
        )

        if epoch % 25 == 0:
            gridfile, s_token_match = ar_sample_batch(epoch, model, batch)
            image_log = wandb.Image(gridfile)
            wandb.log({"image_sample": image_log, "token_match/sample": s_token_match})

        jax.clear_caches()
        gc.collect()

    etime = time.time() - stime
    print(
        f"overfit time for {epochs} epochs -> {etime/60:.4f} mins / {etime/60/60:.4f} hrs"
    )
    return model, train_loss


def jax_collate(batch):
    tokens = jnp.stack([jnp.array(item["input_ids"]) for item in batch], axis=0)
    labels = jnp.stack([jnp.array(item["class_label"]) for item in batch], axis=0)

    return {
        "input_ids": tokens,
        "class_label": labels,
    }


@click.command()
@click.option("-r", "--run", default="overfit")
@click.option("-e", "--epochs", default=10)
@click.option("-bs", "--batch_size", default=config.batch_size)
def main(run, epochs, batch_size):

    dataset = ImageTokenDataset()
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=jax_collate
    )

    ar_model = ART()
    state = nnx.state(ar_model)

    n_params = sum([p.size for p in jax.tree.leaves(state)])
    print(f"model parameters count: {n_params/1e6:.2f}M")

    initial_learning_rate = 3e-4
    end_learning_rate = 3e-5
    power = 1  # setting this to 1 makes it a linear schedule
    schedule = optax.polynomial_schedule(
        init_value=initial_learning_rate,
        end_value=end_learning_rate,
        power=power,
        transition_steps=1000,
    )

    optimizer = nnx.Optimizer(
        ar_model,
        optax.adamw(schedule, b1=0.9, b2=0.995, eps=1e-9, weight_decay=0.001)
    )
    if run == "overfit":
        model, loss = overfit(epochs, ar_model, optimizer, train_loader, schedule)
        wandb.finish()
        print(f"microdit overfitting ended at loss {loss:.4f}")

    # elif run == "train":
    #     trainer(ar_model, optimizer, train_loader, schedule=schedule, epochs=epochs)
    #     wandb.finish()
    #     print("microdit (test) training (on imagenet-1k) in JAX..done")


main()
