import jax
import flax
import optax
import wandb
from jax import numpy as jnp
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import FlaxResNetForImageClassification, AutoImageProcessor
from katara import read_image, wandb_logger
from tqdm.auto import tqdm


image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
resnet_model = FlaxResNetForImageClassification.from_pretrained(
    "microsoft/resnet-50"
).to("cuda")

resnet_model()

resnet_model.config.num_labels = 10
rkey = jax.random.key(333)

# GATE-engine/mini_imagenet, ILSVRC/imagenet-1k
hfdata = load_dataset("uoft-cs/cifar10", split="train")
hfdata = hfdata.take(10000)


def dataproc(batch):
    img = batch["img"]
    img = read_image(img, img_size=32)
    proc_img = image_processor(images=img, return_tensors="np")
    batch["image_array"] = proc_img

    return batch


train_data = hfdata.map(dataproc, batched=True, remove_columns=["img"])


def jax_collate(batch):
    batch = {k: jnp.array([d[k] for k in batch]) for k in batch[0]}
    batch = jax.tree_util.tree_map(jnp.array, batch)

    return batch


train_loader = DataLoader(
    train_data, batch_size=32, shuffle=True, collate_fn=jax_collate
)

optimizer = optax.adam(learning_rate=1e-4)
opt_state = optimizer.init(resnet_model.params)


def loss_compute(logits: jax.Array, labels: jax.Array) -> jax.Array:
    onehot_labels = jax.nn.one_hot(labels, num_classes=labels.shape[-1])
    loss = optax.softmax_cross_entropy(logits, onehot_labels).mean()

    return loss


def loss_func(
    params: jax.Array, inputs: jax.Array, labels: jax.Array, dropout_key
) -> jax.Array:
    model_outputs = resnet_model(
        pixel_values=inputs, params=params, train=True, dropout_key=dropout_key
    )

    model_logits = model_outputs.logits
    loss = loss_compute(model_logits, labels)

    return loss


@jax.jit
def train_step(params, batch, key) -> jax.Array:
    images, labels = batch["image_array"], batch["label"]

    loss, grads = jax.value_and_grad(loss_func)(params, images, labels, key)

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss


def trainer(
    model: FlaxResNetForImageClassification = resnet_model,
    train_loader=train_loader,
    epochs=10,
):
    wandb_logger(
        key=None, model=model, project_name="play_jax", run_name="resnet-tune-1"
    )
    for epoch in tqdm(range(epochs)):
        print(f"training epoch {epoch+1}")

        for step, batch in tqdm(enumerate(train_loader)):
            key, subkey = jax.random.split(rkey)
            model.params, opt_state, train_loss = train_step(
                model.params, opt_state, batch, subkey
            )
            print(f"step {step}: loss => {train_loss}")

        print(f"epoch {epoch+1}, train_loss: {train_loss}")


trainer()
wandb.finish()
