import jax, math, optax
from jax import numpy as jnp, random as jrand
from jax.sharding import NamedSharding as NS, Mesh, PartitionSpec as PS
from jax.experimental import mesh_utils
from flax import nnx

import wandb, click, time, gc, os
from tqdm.auto import tqdm
from functools import partial
from torch.utils.data import DataLoader

from .tinylm import Transformer, TextData 
from .utils import initialize_state, wandb_logger


jax.config.update("jax_default_matmul_precision", "bfloat16")
JAX_TRACEBACK_FILTERING = "off"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
XLA_PYTHON_CLIENT_MEM_FRACTION = 0.20
JAX_DEFAULT_MATMUL_PRECISION = "bfloat16"


class config:
    depth = 12
    attn_heads = 8
    hidden_dim = 768
    mlp_dim = hidden_dim * 2
    vocab_size = 8192
    seed = 256
    key = jrand.key(seed)
    split = 10000
    learn_rate = 1e-4


num_devices = jax.device_count()
devices = jax.devices()

print(f"found {num_devices} JAX device(s)")
for device in devices:
    print(f"{device} / ")


mesh_devices = mesh_utils.create_device_mesh((num_devices,))
mesh = Mesh(mesh_devices, axis_names=("data", None))
print(f'{mesh.shape = }')


def jax_collate(batch):
    tokens = jnp.stack(
        [jnp.array(item["input_ids"], dtype=jnp.bfloat16) for item in batch], axis=0
    )
    mask = jnp.stack(
        [jnp.array(item["attention_mask"], dtype=jnp.bfloat16) for item in batch],
        axis=0,
    )

    return {
        "input_ids": tokens,
        "attention_mask": mask,
    }

def modelpass(logits, tokens):

    output = logits[..., :-1, :]
    targets = tokens[..., 1:].squeeze(1)

    output = output.reshape(-1, output.shape[-1])
    targets = targets.reshape(-1)

    output = output.astype(jnp.bfloat16)
    targets = targets.astype(jnp.bfloat16)

    return output, targets


# compile multidevice versions of train/eval/predict step fn.
model = Transformer(n_layers=6, embed_dim=256, rngs=nnx.Rngs(config.seed))
tx_optimizer = optax.adamw(learning_rate=1e-4)

state, state_sharding = initialize_state(model, mesh, tx_optimizer)
data_sharding = NS(mesh, PS("data"))
rep_sharding = NS(mesh, PS())

@partial(
    jax.jit,
    in_shardings=(
        state_sharding,
        data_sharding
    ),  # type: ignore
    out_shardings=(state_sharding, None),  # type: ignore
    # static_argnums=(2, 3),
    donate_argnums=0,
)  # type: ignore
def train_step(model_state, batch):

    def loss_func(state, batch):
        model = nnx.merge(state.graphdef, state)
        
        tokens = batch["input_ids"]
        logits = model(tokens)

        logits = logits.astype(jnp.bfloat16)
        tokens = tokens.astype(jnp.bfloat16)
        output, targets = modelpass(logits, tokens)

        loss = optax.softmax_cross_entropy(output, targets).mean()

        return loss

    grad_fn = jax.value_and_grad(loss_func)
    loss, grads = grad_fn(model_state.params, batch)
    new_state = model_state.apply_gradients(grads=grads)

    return new_state, loss


jit_train_step = jax.jit(
    train_step,
    in_shardings=(
        state_sharding,
        data_sharding,
        None,
    ),  # type: ignore
    out_shardings=(state_sharding),  # type: ignore
    static_argnums=(2, 3),
    donate_argnums=0,
)


def batch_trainer(epochs, model, train_loader):
    train_loss = 0.0
    model.train()

    wandb_logger(key="", project_name="tinylm_jax")

    stime = time.time()

    batch = next(iter(train_loader))
    print("start overfitting.../")

    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model, batch)
        print(f"epoch {epoch+1}/{epochs}, train loss => {train_loss.item():.4f}")
        wandb.log({"loss": train_loss, "log_loss": math.log10(train_loss)})

        if epoch % 50 == 0:
            pass

        jax.clear_caches()
        jax.clear_backends()
        gc.collect()

    etime = time.time() - stime
    print(
        f"overfit time for {epochs} epochs -> {etime/60:.4f} mins / {etime/60/60:.4f} hrs"
    )

    return model, train_loss


def trainer(epochs, model, optimizer, train_loader):
    train_loss = 0.0
    model.train()

    wandb_logger(key="", project_name="tinylm_jax")
    stime = time.time()

    print("start overfitting.../")

    for epoch in tqdm(range(epochs)):
        for step, batch in tqdm(enumerate(train_loader)):
            train_loss = train_step(model, optimizer, batch)
            print(f"step {step}/{len(train_loader)}, train loss => {train_loss:.4f}")
            wandb.log({"loss": train_loss, "log_loss": math.log10(train_loss)})

            if step % 200 == 0:
                pass

            jax.clear_caches()
            jax.clear_backends()
            gc.collect()

        print(f"epoch {epoch+1}/{epochs}, train loss => {train_loss:.4f}")

    etime = time.time() - stime
    wandb.log({'train/time_mins': etime/60})
    print(f"train time for {epochs} epochs -> {etime/60/60:.4f} hrs")


@click.command()
@click.option("-r", "--run", default="single_batch")
@click.option("-e", "--epochs", default=30)
@click.option("-bs", "--batch_size", default=32)
def main(run, epochs, batch_size):
    embed_dim = config.hidden_dim
    depth = config.depth

    textdata = TextData()
    train_loader = DataLoader(textdata, batch_size=batch_size, collate_fn=jax_collate)

    vs = next(iter(train_loader))
    print(f"sample token shape: {vs['input_ids'].shape}")

    model = Transformer(depth, embed_dim, rngs=nnx.Rngs(config.seed))

    testout = model(jnp.ones((1, 64, embed_dim)))
    print(f"test output shape => {testout.shape}")

    n_params = sum([p.size for p in jax.tree.leaves(nnx.state(model))])
    print(f"model parameters count: {n_params/1e6:.2f}M, ")

    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=config.learn_rate))

    state = nnx.state((model, optimizer))
    state = jax.device_put(state, jax.devices()[0])
    nnx.update((model, optimizer), state)

    if run == "single_batch":
        model, loss = batch_trainer(
            epochs, model=model, train_loader=train_loader
        )
        wandb.finish()
        print(f"single batch training ended at loss: {loss:.4f}")

    elif run == "train":
        print(f"you missed your train looop impl boy")


main()
