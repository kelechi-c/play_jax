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
import optax
from tqdm import tqdm

class config:
    vocab_size: int = 64 * (68000 // 64)
    depth: int = 12
    n_head: int = 12
    n_embd: int = 768
    num_classes: int = 1000
    wte_init_std: float = 0.02
    v_residual: bool = False
    seed = 333
    dtype = jnp.bfloat16

randkey = jrand.key(config.seed)
rngs = nnx.Rngs(config.seed)
xavier_init = nnx.initializers.xavier_uniform()
zero_init = nnx.initializers.constant(0)
one_init = nnx.initializers.constant(1)
normal_init = nnx.initializers.normal(0.02)
trunc_init = nnx.initializers.truncated_normal(0.02)


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

class CausalAttention(nnx.Module):
    def __init__(self, config=config):
        dim = config.n_embd
        heads = config.n_head
        
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.q_linear = nnx.Linear(
            dim, dim, rngs=rngs, kernel_init=xavier_init, use_bias=False
        )
        self.k_linear = nnx.Linear(
            dim, dim, rngs=rngs,
            kernel_init=xavier_init, use_bias=False
        )
        self.v_linear = nnx.Linear(
            dim, dim, rngs=rngs, 
            kernel_init=xavier_init, use_bias=False
        )
        self.out_proj = nnx.Linear(
            dim, dim, rngs=rngs, 
            kernel_init=xavier_init, use_bias=False
        )

        self.lamb1 = nnx.Param(jnp.array(0.5))
        self.lamb2 = nnx.Param(jnp.array(0.5))
        self.rms_norm = nnx.RMSNorm(dim, rngs=rngs)

    def __call__(self, x: Array, attn_mask=None, kv_cache=None, freq=None, v1=None):
        N, T, D = x.shape

        q = self.q_linear(x).reshape(N, T, self.heads, self.head_dim)
        k = self.k_linear(x).reshape(N, T, self.heads, self.head_dim)
        v = self.v_linear(x).reshape(N, T, self.heads, self.head_dim)
        cos, sin = freq

        if v1 is None:
            v1 = v

        v = self.lamb1.value * v + self.lamb2.value * v1.reshape(v.shape)

        if attn_mask is None:
            attn_mask = jnp.tril(jnp.ones((T, T)), k=1).astype(x.dtype)

        new_kv_cache = None
        if kv_cache is not None:
            k_cache, v_cache = kv_cache

            q, k = apply_rotary_embed(q, cos, sin), apply_rotary_embed(k, cos, sin)
            q, k = self.rms_norm(q), self.rms_norm(k)

            if k_cache is not None:
                if isinstance(k_cache, int):
                    k_cache = k
                    v_cache = v
                else:
                    k = jnp.concat([k_cache, k], axis=1)
                    v = jnp.concat([v_cache, v], axis=1)

                new_kv_cache = (k, v)

            y = nnx.dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), mask=attn_mask
            )
            y = y.transpose(1, 2).reshape(x.shape)
            y = self.out_proj(y)

            return (y, v1), new_kv_cache


class MLP(nnx.Module):
    def __init__(self, config=config):
        super().__init__()
        mlp_ratio = 4
        hidden = int(mlp_ratio * config.n_embd)

        self.linear_1 = nnx.Linear(
            config.n_embd, hidden, rngs=rngs, kernel_init=zero_init, use_bias=False
        )
        self.linear_2 = nnx.Linear(
            hidden, config.n_embd, rngs=rngs, kernel_init=zero_init, use_bias=False
        )

    def __call__(self, x: Array) -> Array:
        x = nnx.relu(self.linear_1(x))
        x = self.linear_2(jnp.square(x))

        return x


class FeedForward(nnx.Module):
    def __init__(self, embed_dim, mlp_ratio=4):
        super().__init__()
        hidden = int(mlp_ratio * embed_dim)

        self.w_1 = nnx.Linear(
            embed_dim, hidden, rngs=rngs, kernel_init=trunc_init, bias_init=zero_init
        )
        self.w_2 = nnx.Linear(
            embed_dim, hidden, rngs=rngs, kernel_init=xavier_init, bias_init=zero_init
        )
        self.w_3 = nnx.Linear(
            hidden, embed_dim, rngs=rngs, kernel_init=xavier_init, bias_init=zero_init
        )

    def __call__(self, x: Array) -> Array:
        x = self.w_2(x) * nnx.silu(self.w_1(x))
        x = self.w_3(x)
        return x

class DecoderBlock(nnx.Module):
    def __init__(self, config=config):
        self.attn = CausalAttention(config)
        self.mlp = MLP()
        self.rms_norm = nnx.RMSNorm(config.n_embd, rngs=rngs)

    def __call__(self, x, kv_cache=None, freq=None, v1=None):
        (attn_out, v1), new_kv_cache = self.attn(self.rms_norm(x))
        x = x + attn_out
        x = x + self.mlp(self.rms_norm(x))
        
        return (x, v1), new_kv_cache


class ART(nnx.Module):
    def __init__(self, config=config):
        self.layers = [DecoderBlock() for _ in range(config.depth)]
        self.transformer = nnx.Sequential(*self.layers)

        self.token_embed = nnx.Embed(
            config.vocab_size, config.n_embd, rngs=rngs, 
            embedding_init=normal_init
        )
        self.linear_head = nnx.Linear(
            config.n_embd,
            config.vocab_size,
            rngs=rngs,
            kernel_init=zero_init,
            use_bias=False,
        )
        self.rotary = Rotary(config.n_embd // (2*config.n_head))
        self.rms_norm = nnx.RMSNorm(config.n_embd, rngs=rngs)

    def __call__(self, token_idx, cond_labels):
        b, t = token_idx.shape
        c_tokens = cond_labels + 65536
        targets = token_idx
        token_idx = token_idx[:, :-1]
        token_seq = jnp.concat([c_tokens[None], token_idx], axis=1)

        freq = self.rotary(None, hw=(32, 32))

        x = self.token_embed(token_seq)        
        x = x + x[:, 0:1, :]
        x = self.rms_norm(x)

        v1 = None
        for block in self.layers:
            x, v1 = block(x, freq=freq, v1=v1)[0]

        x = self.rms_norm(x)
        logits = self.linear_head(x).astype(config.dtype)

        loss = optax.softmax_cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1)
        )

        return logits, loss

    def generate(self, class_labels: Array, max_tokens=1024, temp=1.0, cfg_scale=5.0, topk=None):
        b = class_labels.shape[0]
        class_labels = class_labels.repeat(2)
        class_labels[b:] = 1000
        x = (class_labels + 65536)[:, None]
        kv_caches = [(0,0) * len(self.layers)]
        x_init = self.token_embed(x)
        
        freq = self.rotary(None, hw=(32, 32))
        cos, sin = freq
        x_all = x

        for k in tqdm(range(max_tokens)):
            x_emb = self.token_embed(x_all[:, -1:])
            x_emb = self.rms_norm(x_emb)
            cos_local = cos[:, k: k+1, :, :]
            sin_local = sin[:, k : k + 1, :, :]

            freq_local = (cos_local, sin_local)
            v1 = None
            
            for n, layer in enumerate(self.layers):
                (x_emb, v1), new_kv_cache = layer(x_emb, kv_caches[n], freq=freq_local, v1=v1)
                kv_caches[n] = new_kv_cache
                
            x_emb = self.rms_norm(x_emb)
            logits = self.linear_head(x_emb)
            
            logits_cond = logits[:b, :]
            logits_uncond = logits[b:, :]
            logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
            logits = logits / temp
            
            if topk is not None:
                v, _ = jax.lax.top_k(logits, min(topk, logits.shape[-1]))
                logits[logits < v[:,[-1]]] = jnp.inf
                
            probs = nnx.softmax(logits.squeeze(1), axis=-1)
            idx_next = jrand.categorical(randkey, probs, axis=1)
            idx_next = jnp.tile(idx_next, (2, 1))
            x_all = jnp.concat([x_all, idx_next], axis=1)
            
        return x_all[:, 1:]
    