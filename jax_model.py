# image classification with JAX

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

x = jnp.array([[1, 2, 3], [4, 5, 6]])
y = jnp.dot(x, x.T)

# init random image like array
rand_x = np.random.normal(size=(16, 3, 16, 16))
rand_x = jnp.array(rand_x)

rand_x.shape


class Classifier(nnx.Module):
    def __init__(self, rng: nnx.Rngs) -> None:
        super().__init__()
        self.convnet = nnx.Sequential(
            nnx.Conv(16, 16, kernel_size=3, rngs=rng),
            nnx.Linear(16, 10, rngs=rng),
            nnx.BatchNorm(10, rngs=rng),
        )

    def __call__(self, x):
        x = self.convnet(x)
        x = nnx.relu(x)
        x = nnx.softmax(x)

        return x


model = Classifier(rng=nnx.Rngs(params=0))  # model instance
pred = model(rand_x)

nnx.display(model)  # visualizewiht penzai
