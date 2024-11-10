import jax, math
from jax import (
    Array,
    numpy as jnp,
    random as jrand
)
from flax import nnx
from tqdm.auto import tqdm
from typing import List


randkey = jrand.key(3)

# rectifed flow forward pass, loss, and smapling
class RectFlow:
    def __init__(self, model: nnx.Module, sigln: bool = True):
        self.model = model
        self.sigln = sigln

    def __call__(self, x_input: Array, cond: Array) -> Array:

        b_size = x_input.shape[0] # batch_sie

        if self.sigln:
            rand = jrand.normal(randkey, (b_size,)).to_device(x_input.device)
            rand_t = nnx.sigmoid(rand)

        else:
            rand_t = jrand.normal(randkey, (b_size,)).to_device(x_input.device)

        inshape = [1] * len(x_input.shape[1:])
        texp = rand_t.reshape([b_size, *(inshape)])

        z_noise = jrand.normal(randkey, x_input.shape) # input noise with same dim as image 
        z_noise_t = (1 - texp) * x_input + texp * z_noise
        
        v_thetha = self.model(z_noise_t, rand_t, cond)
        
        mean_dim = list(range(1, len(x_input.shape))) # across all dimensions except the batch dim
        mean_square = ((z_noise - x_input - v_thetha) ** 2) # squared difference
        batchwise_mse_loss = jnp.mean(mean_square, axis=mean_dim) # mean loss
        
        return jnp.mean(batchwise_mse_loss)
    
    def sample(self, input_noise: jax.Array, cond, zero_cond=None, sample_steps: int=50, cfg=2.0) -> List[jax.Array]:
        
        batch_size = input_noise.shape[0]
        
        # array reciprocal of sampling steps
        d_steps = 1.0 / sample_steps
        
        d_steps = jnp.array([d_steps] * batch_size).to_device(input_noise.device)
        steps_dim = [1] * len(input_noise.shape[1:])
        d_steps = d_steps.reshape([batch_size], *steps_dim)
        
        images = [input_noise] # noise sequence
        
        for t_step in tqdm(range(sample_steps)):
            
            genstep = t_step / sample_steps # current step
            
            genstep_batched = jnp.array([genstep] * batch_size).to_device(input_noise.device)
            
            cond_output = self.model(input_noise, genstep_batched, cond) # get model output for step
            
            if zero_cond is not None:
                # output for zero conditioning
                uncond_output = self.model(input_noise, genstep_batched, zero_cond)
                cond_output = uncond_output + cfg * (cond_output - uncond_output)
            
            out_noise = input_noise - d_steps * cond_output
            
            images.append(out_noise)
            
        return images