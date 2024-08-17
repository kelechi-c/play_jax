import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx, linen as lnn
from flax.training import train_state

from configs import config
from image_data import train_loader
from tqdm.auto import tqdm

# Init
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)
lr = config.lr
seed = 0

training_loss, testing_loss = [], []
training_accuracy, testing_accuracy = [], []

# Flax model
class Convnet(lnn.module):

    @lnn.compact
    def __call__(self, img):
        x = lnn.Conv(features=32, kernel_size=(3, 3))(img)
        x = lnn.relu(x)
        x = lnn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = lnn.Conv(features=64, kernel_size=(3, 3))(x)
        x = lnn.relu(x)
        x = lnn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1)) #THis is 4 flatten
        x = lnn.Dense(features=256)(x)
        x = lnn.relu(x)
        x = lnn.Dense(features=10)(x)

        return x


def cross_entropy(*, logits, labels):
    encoded_labels = jax.nn.one_hot(labels, num_classes=38)
    ce_loss = optax.softmax_cross_entropy(logits=logits, labels=encoded_labels)

    return ce_loss.mean()


def compute_model_metrics(*, logits, labels):
    loss = cross_entropy(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)

    metrics = {"loss": loss, "accuracy": accuracy}

    return metrics


def compute_loss(params, images, labels):
    logits = Convnet().apply({"params": params}, images)
    loss = cross_entropy(logits, labels)

    return loss


def init_train_state(rng, lr=config.learn_rate):
    convnet = Convnet()
    params = convnet.init(rng, jnp.ones([1, config.image_size, config.image_size, 3]))["params"]
    tx = optax.adam(learning_rate=lr)

    return train_state.TrainState.create(apply_fn=convnet.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch):
    images, labels = batch
    (_, logits), grads = jax.value_and_grad(compute_loss, has_aux=True)(
        state.params, images, labels
    )
    state = state.apply_gradients(grads=grads)
    metrics = compute_model_metrics(logits=logits, labels=labels)

    return state, metrics


@jax.jit
def eval_step(state, batch):
    images, labels = batch
    logits = Convnet().apply({"params": state.params}, images)

    return compute_model_metrics(logits=logits, labels=labels)


def evaluate_model(state, batch):
    test_imgs, test_labels = batch
    metrics = eval_step(state, test_imgs, test_labels)
    metrics = jax.device_get(metrics)
    metrics = jax.tree_map(lambda x: x.item(), metrics)

    return metrics

state = init_train_state(init_rng, lr)


def train_model(state, train_loader, num_epochs=30):
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        train_batch_loss, train_batch_accuracy = [], []
        val_batch_loss, val_batch_accuracy = [], []

        for train_batch in train_loader:
            state, train_metrics = train_step(state, train_batch)
            train_batch_loss.append(train_metrics["loss"])
            train_batch_accuracy.append(train_metrics["accuracy"])

        #         for val_batch in test_loader:
        #             test_metrics = eval_step(state, val_batch)

        #             val_batch_loss.append(test_metrics['loss'])
        #             val_batch_accuracy.append(test_metrics['accuracy'])

        # Loss for the current epoch
        epoch_train_loss = np.mean(train_batch_loss)
        epoch_val_loss = np.mean(val_batch_loss)

        # Accuracy for the current epoch
        epoch_train_acc = np.mean(train_batch_accuracy)
        epoch_val_acc = np.mean(val_batch_accuracy)

        testing_loss.append(epoch_val_loss)
        testing_accuracy.append(epoch_val_acc)

        training_loss.append(epoch_train_loss)
        training_accuracy.append(epoch_train_acc)

        print(
            f"Epoch: {epoch + 1}, loss: {epoch_train_loss:.2f}, acc: {epoch_train_acc:.2f} val loss: {epoch_val_loss:.2f} val acc {epoch_val_acc:.2f} "
        )

    return state


training = train_model(state, train_loader)
