
import jax, jax.numpy as jnp
from jax.sharding import NamedSharding, Mesh, PartitionSpec as PS
from jax.experimental import mesh_utils
from jax import Array, random as jrand

from flax import nnx
import optax, wandb, os, numpy as np
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from tqdm.auto import tqdm
import flax.traverse_util
from flax.serialization import to_state_dict
from functools import partial

device_count = jax.device_count()
devices = jax.local_devices()
JAX_ENABLE_X64=True


device_count = jax.local_device_count()
mesh = Mesh(
  mesh_utils.create_device_mesh((device_count,)), ('data',)
)

model_sharding = NamedSharding(mesh, PS())
data_sharding = NamedSharding(mesh, PS('data'))

split: int = 10000
hfdata = load_dataset('uoft-cs/cifar10', split='train', streaming=True).take(split)


def read_image(img):
    img = np.array(img)
    img = img / 255.0

    return img


class ImageData(IterableDataset):
    def __init__(self, dataset=hfdata):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return split

    def __iter__(self):
        for sample in self.dataset:
            image = sample["img"]
            image = read_image(image)

            image = jnp.array(image)
            label = jnp.array(sample["label"])

            yield image, label # sample["label"]#label


def jax_collate(batch):
    images, labels = zip(*batch)
    batch = (jnp.array(images), jnp.array(labels))
    batch = jax.tree_util.tree_map(jnp.array, batch)
    return batch

traindata = ImageData()
train_loader = DataLoader(traindata, batch_size=128, collate_fn=jax_collate)

xc = next(iter(train_loader))


class MobileBlock(nnx.Module):
    def __init__(self, inchan, outchan, rngs: nnx.Rngs, stride=1):
        self.depthwise_conv = nnx.Sequential(
            nnx.Conv(
                inchan, inchan, kernel_size=(3,3),
                strides=stride, padding=1, feature_group_count=inchan, rngs=rngs
            ),
            nnx.BatchNorm(inchan, rngs=rngs)
        )
        self.pointwise = nnx.Sequential(
            nnx.Conv(inchan, outchan, kernel_size=(1, 1), strides=1, padding=0, rngs=rngs),
            nnx.BatchNorm(outchan, rngs=rngs)
        )

    def __call__(self, x_img):
        x = nnx.relu(self.depthwise_conv(x_img))
        x = nnx.relu(self.pointwise(x))

        return x



class JaxMobilenet(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, num_classes=10):
        self.inputconv = nnx.Conv(3, 32, kernel_size=3, strides=2, padding=1, rngs=rngs)
        self.batchnorm = nnx.BatchNorm(32, rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(1, 1), strides=(1, 1))

        self.convlayer = nnx.Sequential(
            MobileBlock(32, 64, rngs=rngs),
            MobileBlock(64, 128, stride=2, rngs=rngs),
            MobileBlock(128, 256, stride=2, rngs=rngs),
            MobileBlock(256, 512, stride=2, rngs=rngs),
            MobileBlock(512, 512, rngs=rngs),
            MobileBlock(512, 512, rngs=rngs),
            MobileBlock(512, 512, rngs=rngs),
            MobileBlock(512, 1024, stride=2, rngs=rngs),
        )

        self.linear_fc = nnx.Linear(2048, num_classes, rngs=rngs)

    def __call__(self, x_img: jax.Array):
        # x_img = x_img.squeeze(0)
        x = self.batchnorm(self.inputconv(x_img))
        x = nnx.relu(x)
        x = self.convlayer(x)
        x = self.avg_pool(x)

        x = x.reshape((x.shape[0], -1))

        x = self.linear_fc(x)

        return nnx.softmax(x, axis=1)

cnn_model = JaxMobilenet(rngs=nnx.Rngs(0))

optimizer = nnx.Optimizer(cnn_model, optax.adamw(learning_rate=1e-3))

s = cnn_model(xc[0])
print(s.shape)

n_params = sum([p.size for p in jax.tree.leaves(nnx.state(cnn_model))])
print(f"number of parameters: {n_params/1e6:.3f}M")
# replicate model
state = nnx.state((cnn_model, optimizer))
state = jax.device_put(state, model_sharding)
nnx.update((cnn_model, optimizer), state)

# learn_rate = 1e-4

metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average('loss')
)

import pickle
from flax.core import freeze

def save_paramdict_pickle(params, filename='/tmp/model.pkl'):
    # params = nnx.state(model)
    # params = state.filter(nnx.Param)
    params = jax.device_get(params)

    state_dict = to_state_dict(params)
    frozen_state_dict = freeze(state_dict)

    flat_state_dict = flax.traverse_util.flatten_dict(frozen_state_dict, sep='.')

    with open(filename, 'wb') as f:
        pickle.dump(frozen_state_dict, f)

    return flat_state_dict

# def wandb_logger(key: str, model, project_name, run_name):  # wandb logger
#     # initilaize wandb
#     key = ''
#     wandb.login(key=key)
#     wandb.init(project=project_name, name=run_name)
    # wandb.watch(model)

import gc

XLA_PYTHON_CLIENT_MEM_FRACTION = 0.30
gc.collect()
jax.clear_caches()

@nnx.jit
def train_step(image, label, model, optimizer):
    def loss_func(model):
        logits = model(image)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits,
            labels=label
        ).mean()

        return loss

    loss, grads = nnx.value_and_grad(loss_func, allow_int=True)(model)
    optimizer.update(grads)

    return loss, model


def trainer(model=cnn_model, train_loader=train_loader, optimizer=optimizer):
    epochs = 1

    for epoch in tqdm(range(epochs)):
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            image, label = batch
            image, label = jax.device_put((image, label), data_sharding)

            # Perform training step
            train_loss, model = train_step(image, label, model, optimizer)

            # Get mean loss across devices
            loss_value = train_loss.mean().item()
            print(f"step {step}, loss-> {loss_value:.4f}")
        state = nnx.state((model, optimizer))
        state = jax.device_get(state)
        nnx.update((model, optimizer), state)
        print(f'epoch {epoch}/{epochs}')

    # Return the model from the first device (they should all be identical)
    return state

trainer()