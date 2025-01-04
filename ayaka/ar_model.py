import jax, math, optax
from jax import numpy as jnp, random as jrand, Array
from flax import nnx
from einops import rearrange
import numpy as np
from transformers import GPT2Tokenizer, AutoTokenizer

class config:
    vocab_size: int = 64 * (68000 // 64)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    num_classes: int = 1000
    wte_init_std: float = 0.02
    v_residual: bool = False