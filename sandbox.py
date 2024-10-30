# simple jax intro code
import jax
import optax
from flax import nnx
from flax import linen as nn
from jax import numpy as jnp

img_size = 28
batch_size = 32
steps = 1000
num_classes = 10


class TinyCNN(nn.Module):
    features = 32

    @nn.compact
    def __call__(self, x_img: jax.Array):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x_img)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=10)(x)

        return x


model = TinyCNN()


@jax.jit
def apply_model(state, images, labels):
    def loss_fn(params):
        model_logits = state.apply_fn({"params": params}, images)
        one_hot = jax.nn.one_hot(labels, num_classes=num_classes)
        loss = jnp.mean(
            optax.softmax_cross_entropy(logits=model_logits, labels=one_hot)
        )

        return loss, model_logits

    grad_func = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_func(state.params)
    acc = jnp.mean((jnp.argmax(logits, -1) == labels))

    return grads, loss, acc


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)
