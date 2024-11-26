### JAX, my beloved framework :)

I just really love **JAX**, forgive the constant rants about it >_<.
I initially liked it, as a **Deepmind/Midjourney** fan and all, and the code looked **beautiful**.\

But after using it personally and seeing how good it was, the control, the synergy with **TPUs**,
the ease of distributed training/sharding, how adaptable it was, etc, I fell in love more.

Yes, it is not widely used as Pytorch(this even makes it feel better), but it is flexible and all.

Also, for stuff I wanna train with **Google's TRC grants**, I will need JAX, def not Pytorch on TPUs
(JAX is good on both btw).

For neural networks, I prefer using the **nnx** API, so my exmaples will work with it. 


##### stuff I ported to Jax
- tiny GPT/transformer
- Mobilenet(CNN)
- Rectified flow (sampling/loss)
- DiT/MicroDiT


### stuff/snippets I learnt/used

1. data parallelism with JAX/NNX
Plug in the following code.

* define device mesh and sharding for data/model
```python
import jax
from jax.sharding import NamedSharding, Mesh, PartitionSpec as PS
from jax.experimental import mesh_utils

device_count = jax.device_count()
devices = jax.local_devices()

device_count = jax.local_device_count()
mesh = Mesh(mesh_utils.create_device_mesh((device_count,)), ('data',))

model_sharding = NamedSharding(mesh, PS())
data_sharding = NamedSharding(mesh, PS('data'))
```
* replicate the model params/optimizer on all devices
```python
# replicate model
state = nnx.state((cnn_model, optimizer))
state = jax.device_put(state, model_sharding)
nnx.update((cnn_model, optimizer), state)
```

* shard the data batch across the TPU devices
```python
image, label = batch
image, label = jax.device_put((image, label), data_sharding)
...

```
* retrieve and merge state
```python
state = nnx.state((model, optimizer))
state = jax.device_get(state)
nnx.update((model, optimizer), state)
```

2. Saving the model checkpoint in a single file
```python
import jax, pickle
from flax import nnx
from flax.core import freeze
import flax.traverse_util
from flax.serialization import to_state_dict

def save_paramdict_pickle(params, filename='model.pkl'):
    params = nnx.state(model)
    params = jax.device_get(params)

    state_dict = to_state_dict(params)
    frozen_state_dict = freeze(state_dict)

    flat_state_dict = flax.traverse_util.flatten_dict(frozen_state_dict, sep='.')

    with open(filename, 'wb') as f:
        pickle.dump(flat_state_dict, f)

    return flat_state_dict

```