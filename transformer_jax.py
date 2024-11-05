import jax, math
from jax import numpy as jnp
from flax.experimental import nnx
from einops import rearrange

attn_heads = 6
embed_dim = 768
vocab_size = 32000
hidden_size = 1024
n_layers = 12


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
        print(f"attn logits shape => {attn_logits.shape}")

        attn_score = nnx.softmax(attn_logits, dim=1)
        attn_output = attn_score @ v

        output = rearrange(attn_output, 'b h l d -> b l (h d)')
        output = self.dropout(self.outproject(output))
        print(f"MHSA out shape => {output.shape}")

        return output


class MLP(nnx.Module):
    def __init__(self, hidden_size, rngs: nnx.Rngs):
        super().__init__()
        self.layernorm = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.linear1 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)

    def forward(self, x_input: jax.Array) -> jax.Array:
        x = self.layernorm(x_input)
        x = nnx.silu(self.linear1(x))
        x = self.linear2(x)

        return x

class DecoderBlock(nnx.Module):
    def __init__(self, embed_dim, rngs: nnx.Rngs, hidden_size=1024):
        super().__init__()
        self.layernorm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.attention = CausalSelfAttention(rngs=rngs)
        self.ffn_layer = MLP(hidden_size=hidden_size, rngs=rngs)

    def forward(self, x_token: jax.Array) -> jax.Array:
        x = x_token + self.layernorm(self.attention(x_token))
        x = x + self.layernorm(self.ffn_layer(x))

        return x
    
# autoregressive transformer block, or just an SLM
class Transformer(nnx.Module):
    def __init__(self, n_layers, embed_dim, rngs: nnx.Rngs, hidden_size=1024, vocab_size=32000):
        super().__init__()
        self.wt_embed = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.wp_embed = nnx.Embed(hidden_size, embed_dim, rngs=rngs)
        self.layernorm = nnx.LayerNorm(embed_dim, rngs=rngs)
        decoder_layers = [DecoderBlock(rngs=rngs) for _ in range(n_layers)]
        self.decoder_layers = nnx.Sequential(**decoder_layers)
        self.lm_head = nnx.Linear(embed_dim, vocab_size, rngs=rngs)

    def forward(self, x_tokens: jax.Array) -> jax.Array:
        x = self.layernorm(self.decoder_layers(x_tokens))
        print(f'decoder out shape => {x.shape}')
        x = self.lm_head(x)
        output = nnx.softmax(x, axis=1)
        print(f"lm/softmax out shape => {x.shape}")

        return output


    def generate(self, cond_token, max_outlen=256, temperature=0.1):
        pass


slm_model = Transformer(12, 768, rngs=nnx.Rngs(3))

graph, params, _ = nnx.split(slm_model, nnx.Param)
n_params = sum([p.size for p in jax.tree_leaves(params)])
print(f'number of parameters: {n_params}')