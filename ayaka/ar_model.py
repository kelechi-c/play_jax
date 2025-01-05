"""
Implementation of an Autoregressive Image generation model in JAX/Flax,
adapted from https://github.com/fal-ai/diffusion-speedrun
"""

import jax, math
from jax import numpy as jnp, random as jrand, Array
from flax import nnx
from einops import rearrange
import numpy as np
from cosmos_tokenizer.image_lib import ImageTokenizer

class config:
    vocab_size: int = 64 * (68000 // 64)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    num_classes: int = 1000
    wte_init_std: float = 0.02
    v_residual: bool = False

class Rotary(nnx.Module):
    def __init__(self, dim, base=100, h=128, w=128):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2)).float() / (dim))
        self.h = h
        self.w = w
        
        # t_h = 
    def __call__(self, x, hw=None):
        pass


def apply_rotary_embed(x, cos, sin):
    cos, sin = cos[:, :x.shape[1]], sin[:, :x.shape[1]]
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 *(-sin) + x2 * cos
    
    return jnp.concat([y1, y2], axis=3).astype(x.dtype)
