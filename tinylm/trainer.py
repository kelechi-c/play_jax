import jax, math, optax
from jax import numpy as jnp, random as jrand, Array
from jax.sharding import NamedSharding as NS, Mesh, PartitionSpec as PS
from jax.experimental import mesh_utils
from flax import nnx

import wandb, click, time, gc, os
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader, IterableDataset

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
mesh = Mesh(mesh_devices, axis_names="data")
data_sharding = NS(mesh, PS("data"))
rep_sharding =NS(mesh, PS()) 


def wandb_logger(key: str, project_name, run_name=None):  # wandb logger
    # initilaize wandb
    wandb.login(key=key)
    wandb.init(project=project_name, name=run_name or None)


def modelpass(logits, tokens):

    output = logits[..., :-1, :]
    targets = tokens[..., 1:].squeeze(1)

    output = output.reshape(-1, output.shape[-1])
    targets = targets.reshape(-1)

    output = output.astype(jnp.bfloat16)
    targets = targets.astype(jnp.bfloat16)

    return output, targets


def loss_func(model, batch):
    tokens = batch["input_ids"]
    logits = model(tokens)

    logits = logits.astype(jnp.float32)
    tokens = tokens.astype(jnp.float32)

    output, targets = modelpass(logits, tokens)

    loss = optax.softmax_cross_entropy(output, targets).mean()

    return loss, logits


@nnx.jit
def train_step(model, optimizer, batch):
    gradfn = nnx.value_and_grad(loss_func, has_aux=True)
    (loss, logits), grads = gradfn(model, batch)
    optimizer.update(grads)

    return loss


def batch_trainer(epochs, model, optimizer, train_loader):
    train_loss = 0.0
    model.train()

    wandb_logger(key="", project_name="tinylm_jax")

    stime = time.time()

    batch = next(iter(train_loader))
    print("start overfitting.../")

    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model, optimizer, batch)
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
            print(f"epoch {epoch+1}/{epochs}, train loss => {train_loss:.4f}")
            wandb.log({"loss": train_loss, "log_loss": math.log10(train_loss)})

            if epoch % 200 == 0:
                pass

            jax.clear_caches()
            jax.clear_backends()
            gc.collect()

    etime = time.time() - stime
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
            epochs, model=model, optimizer=optimizer, train_loader=train_loader
        )
        wandb.finish()
        print(f"single batch training ended at loss: {loss:.4f}")

    elif run == "train":
        print(f"you missed your train looop impl boy")


main()
