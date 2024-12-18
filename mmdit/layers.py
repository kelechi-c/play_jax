import jax, dataclasses, math
from jax import (
    Array,
    numpy as jnp,
    random as jrand
)
from flax import nnx
from einops import rearrange

@dataclasses
class config:
    embed_dim: int = 768
    img_size: int = 256
    patch_size: int = 4

rkey = jrand.key(3)

# input patchify layer, 2D image to patches
class PatchEmbed(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        patch_size: int=4,
        img_size: int=256,
        in_chan: int=3,
        embed_dim: int=768
    ):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.gridsize = tuple([s//p for s, p in zip(img_size, patch_size)])
        self.num_patches = self.gridsize[0] * self.gridsize[1]
        
        self.conv_project = nnx.Conv(
            in_chan,
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size
        )
    def __call__(self, img: Array) -> Array:
        x = self.conv_project(img)
        x = x.flatten()
        
        return x


# modulation with shift and scale
def modulate(x_array: Array, shift, scale) -> Array:
    x = x_array * scale.unsqueeze(1)
    x = x + shift.unsqueeze(1)
    
    return x


# embeds a flat vector
class VectorEmbedder(nnx.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.linear_1 = nnx.Linear(input_dim, hidden_size)
        self.linear_2 = nnx.Linear(hidden_size, hidden_size)
        
    def __call__(self, x: Array):
        x = nnx.silu(self.linear_1(x))
        x = self.linear_2(x)
        
        return x

# self attention block
class SelfAttention(nnx.Module):
    def __init__(self, attn_heads, embed_dim, rngs: nnx.Rngs, drop=0.1):
        super().__init__()
        self.attn_heads = attn_heads
        self.head_dim = embed_dim // attn_heads

        self.q_linear = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.k_linear = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.v_linear = nnx.Linear(embed_dim, embed_dim, rngs=rngs)

        self.outproject = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.dropout = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x_input: jax.Array) -> jax.Array:
        q = self.q_linear(x_input)
        k = self.k_linear(x_input)
        v = self.v_linear(x_input)

        q, k, v = map(lambda x: rearrange(x, "b l (h d) -> b h l d", h=self.attn_heads), (q, k, v))
        
        qk = q @ jnp.matrix_transpose(k)
        attn_logits = qk / math.sqrt(self.head_dim)  # attention computation

        attn_score = nnx.softmax(attn_logits, axis=-1)
        attn_output = attn_score @ v

        output = rearrange(attn_output, "b h l d -> b l (h d)")
        output = self.dropout(self.outproject(output))
        print(f'attn out shape => {output.shape}')
        return output

class SwigluFFN(nnx.Module):
    def __init__(self, input_dim, hidden_dim, mlp_ratio):
        super().__init__()
        hidden_dim = int(2 * hidden_dim/3)
        self.lin_1 = nnx.Linear(input_dim, hidden_dim, use_bias=False)
        self.lin_2 = nnx.Linear(hidden_dim, input_dim, use_bias=False)
        self.lin_3 = nnx.Linear(input_dim, hidden_dim, use_bias=False)
        
    def __call__(self, x: Array) -> Array:
        x = self.lin_2(x) * nnx.silu(self.lin_1(x))
        x = self.lin_3(x)
        
        return x


class RMSNorm(nnx.Module):
    def __init__(self, hidden_dim, eps, elementwise_affine):
        super().__init__()
        self.eps = eps
        self.learnable_scale = elementwise_affine
        self.weight = nnx.Param(jnp.empty(hidden_dim))

    def _rmsnorm(self, x: Array) -> Array:
        x = jnp.pow(x, 2).mean(axis=-1, keepdims=True) + self.eps
        x = x * (1 / x**0.5)

        return x

    def __call__(self, x: Array) -> Array:
        x = self._rmsnorm(x)
        x = x * self.weight.to_device(x.device)

        return x

class FinalMLP(nnx.Module):
    def __init__(self, hidden_size, patch_size, out_chan, rngs: nnx.Rngs):
        super().__init__()
        self.layernorm = nnx.LayerNorm(hidden_size, epsilon=1e-6, rngs=rngs)
        self.linear = nnx.Linear(hidden_size, patch_size*patch_size*out_chan, rngs=rngs)
        self.adaln_mod = nnx.Linear(hidden_size, 2*hidden_size, rngs=rngs)

    def __call__(self, x_input: Array, cond: Array) -> Array:
        cond_mod = nnx.silu(self.adaln_mod(cond))
        shift, scale = jnp.split(cond_mod, axis=1)
        x = modulate(self.layernorm(x_input), shift, scale)
        x = self.linear(x)

        return x
    

class MMDiTblock(nnx.Module):
    def __init__(
            self,
            hidden_size,
            attn_heads,
            mlp_ratio, rms_norm, rngs
        ):
        super().__init__()
        if rms_norm:
            self.norm = RMSNorm(hidden_dim=hidden_size, eps=1e-6, elementwise_affine=False) 
        else:
            self.norm = nnx.LayerNorm(hidden_size, epsilon=1e-6)
        
        self.attention = SelfAttention(attn_heads, embed_dim=hidden_size, rngs=rngs)