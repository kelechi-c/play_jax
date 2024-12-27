import jax

print(f'JAX recognized devices:\n local device count = {jax.local_device_count()} \n {jax.local_devices()}')