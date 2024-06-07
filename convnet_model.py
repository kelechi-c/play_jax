import flax
import jax
import jax.numpy as jnp
import optax
from flax import nnx, linen as lnn
from flax.training import train_state


class Convnet(lnn.module):

    @lnn.compact
    def __call__(self, img):
        x = lnn.Conv(features=32, kernel_size=(3, 3))(img)
        x = lnn.relu(x)
        x = lnn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = lnn.Conv(features=64, kernel_size=(3, 3))(x)
        x = lnn.relu(x)
        x = lnn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = lnn.Dense(features=256)(x)
        x = lnn.relu(x)
        x = lnn.Dense(features=10)(x)

        return x
