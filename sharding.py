import jax
from jax.sharding import NamedSharding, Mesh, PartitionSpec as PS
from jax.experimental import mesh_utils
from jax import random as jrand

device_count = jax.device_count() # 8 devices for tpuv2-8
devices = jax.devices()

randtensor = jrand.normal(jrand.key(33), (32, 64, 1024))

mesh_devices = mesh_utils.create_device_mesh((device_count,))
mesh = Mesh(mesh_devices, axis_names="axis")

sharding = NamedSharding(mesh, PS("axis"))

sharded_batch: jax.Array = jax.device_put(randtensor, sharding) # ransfer to device
sharded_batch.sharding

print(f"original batch shape => {randtensor.shape}")  # (32, 64, 1024)
print(f'Sharded shape per device => {sharding.shard_shape(sharded_batch.shape)}')
# (4, 64, 1024) (4x8 gives the orginal batch size of 32 :)