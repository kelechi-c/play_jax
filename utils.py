import jax, optax, wandb
from flax import nnx
from jax import numpy as jnp
from tqdm.auto import tqdm


def jax_collate(batch):
    images, labels = zip(*batch)
    batch = jnp.array(batch)
    batch = jax.tree_util.tree_map(jnp.array, batch)

    return batch


def loss_func(model, batch):
    image, label = batch
    logits = model(image)
    loss = optax.softmax_cross_entropy(logits, labels=label).mean()

    return loss, logits


def wandb_logger(key: str, project_name, run_name):  # wandb logger
    # initilaize wandb
    wandb.login(key=key)
    wandb.init(project=project_name, name=run_name)


@nnx.jit
def train_step(model, optimizer, metrics: nnx.MultiMetric, batch):
    gradfn = nnx.value_and_grad(loss_func, has_aux=True, allow_int=True)
    (loss, logits), grads = gradfn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch[1])
    optimizer.update(grads)

    return loss


def trainer(model, optimizer, train_loader):
    epochs = 10
    train_loss = 0.0
    wandb_logger(
        key=None,
        model=model,
        project_name="play_jax",
        run_name="-tpu",
    )

    for epoch in tqdm(range(epochs)):
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

            train_loss = train_step(model, optimizer, batch)
            print(f"step {step}, loss-> {train_loss.item():.4f}")

            wandb.log({"loss": train_loss.item()})

        print(f"epoch {epoch}, train loss => {train_loss}")
    
    wandb.finish()