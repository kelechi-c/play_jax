import jax, math
from jax import numpy as jnp
from flax.experimental import nnx
from einops import rearrange

attn_heads = 6
embed_dim = 768


class CausalSelfAttention(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, embed_dim: int=embed_dim, attn_heads: int=attn_heads, drop=0.1):
        super().__init__()
        self.attn_heads = attn_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // attn_heads
        
        self.q_linear = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.k_linear = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.v_linear = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        
        self.outproject = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.dropout = nnx.Dropout(drop, rngs=rngs)
        
    def forward(self, x_input: jax.Array) -> jax.Array:
        q = self.q_linear(x_input)
        k = self.k_linear(x_input)
        v = self.v_linear(x_input)
        
        q, k, v = map(rearrange('b l (h d) -> b h l d', h=self.attn_heads), (q, k, v))
        attn_weight = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim) # attention computation
        
        b, l, h, d = q.shape # just getting the shape, hehe
        # causal attention mask
        mask = jnp.tril(jnp.ones_like((l, l)), k=1).astype(x_input.dtype)
        attn_logits = jnp.where(mask == 0, jnp.inf, attn_weight) # attn_weight - jnp.inf * mask[None, None, :, :]
        
        attn_score = nnx.softmax(attn_logits, dim=1)
        attn_output = attn_score @ v
        
        output = rearrange(attn_output, 'b h l d -> b l (h d)')
        output = self.dropout(self.outproject(output))
        
        return output


class MLP(nnx.Module):
    def __init__(self, hidden_size, rngs: nnx.Rngs):
        super().__init__()
        self.layernorm = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.mlp_layer1 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.mlp2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)

    def forward(self, x_input: jax.Array) -> jax.Array:
        x = self.layernorm(x_input)
        x = nnx.silu(self.mlp_layer1(x))
        x = self.mlp2(x)
        
        return x