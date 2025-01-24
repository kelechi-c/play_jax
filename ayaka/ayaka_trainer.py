import jax, flax, math, torch
from jax import numpy as jnp, random as jrand, Array
from flax import nnx
from jax.sharding import NamedSharding as NS, Mesh, PartitionSpec as PS
from jax.experimental import mesh_utils
jax.config.update("jax_default_matmul_precision", "bfloat16")

import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from functools import partial
from flax.training import train_state
import flax.traverse_util, builtins, torch
from flax.serialization import to_state_dict, from_state_dict
from flax.core import freeze, unfreeze
import json, os, wandb, gc, time, pickle
import matplotlib.pyplot as plt
from PIL import Image
from safetensors import safe_open
from cosmos_tokenizer.image_lib import ImageTokenizer
from huggingface_hub import login, snapshot_download

# XLA/JAX flags
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

model_name = "Cosmos-Tokenizer-DI8x8"
hf_repo = f"nvidia/{model_name}"
local_dir = 'checks'
os.makedirs(local_dir, exist_ok=True)
snapshot_download(repo_id=model_name, local_dir=local_dir)
decoder = ImageTokenizer(
    checkpoint_dec=f"{local_dir}/{model_name}/decoder.jit"
)

class ImageTokenDataset(Dataset):
    def __init__(self, safetensor_path="./imagenet_di8x8.safetensors", debug=True):
        self.safetensor_path = safetensor_path
        self.split = 100_000
        
        metadata_path = safetensor_path.replace(".safetensors", "_metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
            self.total_samples = self.metadata["total_samples"]

        if debug:
            self.total_samples = 16

        with safe_open(safetensor_path, framework="flax") as f:
            self.indices = f.get_tensor("indices").to(jnp.uint16).astype(jnp.int64)[:self.split]
            self.labels = f.get_tensor("labels").astype(jnp.int64)[:self.split]

        if debug:
            self.indices = self.indices[:16]
            self.labels = self.labels[:16]

    def __len__(self):
        return int(self.total_samples)

    def __getitem__(self, idx):
        indices = self.indices[idx].reshape(-1)
        # replace randomly with 1000
        fill_cond = jrand.normal(randkey, (indices.shape)) < 0.05
        indices = jnp.where(fill_cond, indices, 1000)
        class_label = self.labels[idx]

        return {"input_ids": indices, "class_label": class_label}


def decode(data):
    data = data.reshape(1, 32, 32)
    data = torch.from_numpy(np.array(data))

    # Decode the image
    with torch.no_grad():
        reconstructed = decoder.decode(data)

    img = ((reconstructed[0].cpu().float() + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    img = img.permute(1, 2, 0).numpy()
    img = Image.fromarray(img)

    return img


def inspect_latents(batch):
    batch = rearrange(batch, "b c h w -> b h w c")
    # print(batch.shape)
    # img_latents = img_latents / config.vaescale_factor
    batch = [process_img(x) for x in batch]
    file = f"images/preproc_cc-by.jpg"
    gridfile = image_grid(batch, file)
    print(f"sample saved @ {gridfile}")


def image_grid(pil_images, file, grid_size=(3, 3), figsize=(4, 2)):
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

def sample_image_batch(step, model, batch):
    labels, shape, ground_latents, captions = (
        batch["text_embedding"][:9],
        batch["vae_latent_shape"][:9],
        batch["vae_latent"][:9],
        batch["caption"][:9],
    )
    # print(shape)
    start_noise = jrand.normal(randkey, (len(labels), shape[1], shape[2], shape[0]))
    print(f"sampling from noise of shape {start_noise.shape} / captions {captions}")

    pred_model = device_get_model(model)
    pred_model.eval()

    image_batch = pred_model.sample(start_noise, labels)

    batch_latents = rearrange(ground_latents, "b c h w -> b h w c")
    v_sample_error = (batch_latents - image_batch) ** 2

    file = f"mdsamples/{step}_microdit.png"
    batch = [process_img(x) for x in image_batch]

    gridfile = image_grid(batch, file)
    print(f"sample saved @ {gridfile}")

    del pred_model
    jax.clear_caches()
    gc.collect()

    v_sample_error = v_sample_error.mean()

    print(f"sample_error {v_sample_error}")

    return gridfile, v_sample_error


def wandb_logger(key: str, project_name, run_name=None):  # wandb logger
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
def save_paramdict_pickle(model, filename="dit_model.ckpt"):
    params = nnx.state(model)
    params = jax.device_get(params)

    state_dict = to_state_dict(params)
    frozen_state_dict = freeze(state_dict)

    flat_state_dict = flax.traverse_util.flatten_dict(frozen_state_dict, sep=".")

    with open(filename, "wb") as f:
        pickle.dump(frozen_state_dict, f)

    return flat_state_dict


def load_paramdict_pickle(model, filename="dit_model.ckpt"):
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
    out_shardings=(None, None),
)
def train_step(model, optimizer, image, text):

    def loss_func(model, img_latents, label):
        img_latents = (
            rearrange(img_latents, "b c h w -> b h w c") * config.vaescale_factor
        )
        loss = model(img_latents, label)

        return loss

    gradfn = nnx.value_and_grad(loss_func)
    loss, grads = gradfn(model, image, text)

    grad_norm = optax.global_norm(grads)

    optimizer.update(grads)

    return loss, grad_norm


def overfit(epochs, model, optimizer, train_loader, schedule):
    train_loss = 0.0
    model.train()

    batch = next(iter(train_loader))

    wandb_logger(key="yourkey", project_name="microdit_overfit")

    stime = time.time()

    print("start overfitting.../")
    for epoch in tqdm(range(epochs)):
        lr = schedule(epoch)

        latent, text_embed = batch["vae_latent"], batch["text_embedding"]
        train_loss, grad_norm = train_step(model, optimizer, latent, text_embed)
        print(
            f"step {epoch}, loss-> {train_loss.item():.4f}, grad_norm {grad_norm.item()}"
        )

        wandb.log(
            {
                "loss": train_loss.item(),
                "log_loss": math.log10(train_loss.item() + 1e-8),
                "grad_norm": grad_norm.item(),
                "log_grad_norm": math.log10(grad_norm.item() + 1e-8),
                "lr": lr,
            }
        )

        if epoch % 25 == 0 and epoch != 0:
            gridfile, sample_error = sample_image_batch(epoch, model, batch)
            image_log = wandb.Image(gridfile)
            wandb.log({"image_sample": image_log, "sample_error": sample_error})

        jax.clear_caches()
        gc.collect()

    etime = time.time() - stime
    print(
        f"overfit time for {epochs} epochs -> {etime/60:.4f} mins / {etime/60/60:.4f} hrs"
    )
    return model, train_loss
