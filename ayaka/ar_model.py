"""
Implementation of an Autoregressive Image generation model in JAX/Flax,
adapted from https://github.com/fal-ai/diffusion-speedrun
"""

import jax, math
from jax import numpy as jnp, random as jrand, Array
from flax import nnx
from einops import rearrange
from cosmos_tokenizer.image_lib import ImageTokenizer
import numpy as np
import optax
from tqdm.auto import tqdm


class config:
    vocab_size: int = 64 * (68000 // 64)
    depth: int = 6
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
        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2)).astype(config.dtype) / (dim))

        self.h = h
        self.w = w

        t_h = jnp.arange(h)
        t_w = jnp.arange(w)

        freqs_h = jnp.outer(t_h, inv_freq)[:, None]
        freqs_w = jnp.outer(t_w, inv_freq)[None]
        freqs_h = jnp.tile(freqs_h, (1, w, 1))
        freqs_w = jnp.tile(freqs_w, (h, 1, 1))
        freqs_hw = jnp.concat([freqs_h, freqs_w], axis=2)

        self.freqs_hw_cos = nnx.Param(jnp.cos(freqs_hw))
        self.freqs_hw_sin = nnx.Param(jnp.sin(freqs_hw))

        self.cache_cos = nnx.Param(jnp.ones((1, 1024, 1, 32)))
        self.cache_sin = nnx.Param(jnp.ones((1, 1024, 1, 32)))

    def __call__(self, x: Array, hw=None, extend_with_register_tokens=0, augment=False):
        if self.cache_cos is not None and self.cache_sin is not None:
            return self.cache_cos, self.cache_sin

        if hw is not None:
            this_h, this_w = hw
        else:
            this_hw = x.shape[1]
            this_h, this_w = int(this_hw**0.5), int(this_hw**0.5)

        if augment:
            start_h = jrand.randint(randkey, (1,), 0, self.h - this_h + 1).item()
            start_w = jrand.randint(randkey, (1,), 0, self.w - this_w + 1).item()
        else:
            start_h = 0
            start_w = 0

        cos = self.freqs_hw_cos[start_h : start_h + this_h, start_w : start_w + this_w]
        sin = self.freqs_hw_cos[start_h : start_h + this_h, start_w : start_w + this_w]

        cos = cos.reshape(this_h * this_w, -1)
        sin = sin.reshape(this_h * this_w, -1)

        if extend_with_register_tokens > 0:
            cos = jnp.concat(
                [
                    jnp.ones(
                        (extend_with_register_tokens, cos.shape[1]), device=cos.device
                    ),
                    cos,
                ],
                axis=0,
            )
            sin = jnp.concat(
                [
                    jnp.zeros(
                        (extend_with_register_tokens, sin.shape[1]), device=cos.device
                    ),
                    sin,
                ],
                axis=0,
            )
        # update cache
        self.cache_cos.value = cos[None, :, None, :]  # 1, N, 1, D
        self.cache_sin.value = sin[None, :, None, :]

        return self.cache_cos.value, self.cache_sin.value


def rms_norm(x, target_rms=1.0, epsilon=1e-8):
    rms = jnp.sqrt(jnp.mean(jnp.square(x)))
    scale_factor = target_rms / (rms + epsilon)
    normalized_x = x * scale_factor
    return normalized_x


def apply_rotary_embed(x: Array, cos: Array, sin: Array):
    cos, sin = cos[:, : x.shape[1]], sin[:, : x.shape[1]]
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos

    return jnp.concat([y1, y2], axis=3).astype(x.dtype)


from einops import rearrange


# causal self attention block
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
            dim, dim, rngs=rngs, kernel_init=xavier_init, use_bias=False
        )
        self.v_linear = nnx.Linear(
            dim, dim, rngs=rngs, kernel_init=xavier_init, use_bias=False
        )
        self.out_proj = nnx.Linear(
            dim, dim, rngs=rngs, kernel_init=xavier_init, use_bias=False
        )

        self.lamb1 = nnx.Param(jnp.array(0.5))
        self.lamb2 = nnx.Param(jnp.array(0.5))
        self.rms_norm = nnx.RMSNorm(dim, rngs=rngs)

    def __call__(
        self, x: Array, attn_mask=None, kv_cache=None, freq: tuple = None, v1=None
    ):
        N, T, D = x.shape
        # print(f"attn xin {x.shape}")

        q = self.q_linear(x)  # .reshape(N, T, self.heads, self.head_dim)
        k = self.k_linear(x)  # .reshape(N, T, self.heads, self.head_dim)
        v = self.v_linear(x)  # .reshape(N, T, self.heads, self.head_dim)

        q, k, v = map(
            lambda x: rearrange(x, "b l (h d) -> b h l d", h=self.heads), (q, k, v)
        )

        cos, sin = freq

        if v1 is None:
            v1 = v

        v = self.lamb1.value * v + self.lamb2.value * v1.reshape(v.shape)
        # print(f"v after residual: v.shape={v.shape}")

        new_kv_cache = None

        if kv_cache is not None:  # KV Cache branch - No mask for generation
            k_cache, v_cache = kv_cache

            q, k = apply_rotary_embed(q, cos, sin), apply_rotary_embed(k, cos, sin)
            # print(f"q, k after rotary: q.shape={q.shape}, k.shape={k.shape}")
            q, k = rms_norm(q), rms_norm(k)
            # print(f"q, k after rms_norm: q.shape={q.shape}, k.shape={k.shape}")

            if k_cache is not None and not isinstance(
                k_cache, int
            ):  # Added check for int initialization
                k = jnp.concat([k_cache, k], axis=1)
                v = jnp.concat([v_cache, v], axis=1)
                # print(f"k, v after cache concat: k.shape={k.shape}, v.shape={v.shape}")

            new_kv_cache = (k, v)

            y = nnx.dot_product_attention(
                q,
                k,
                v,
                # mask=attn_mask, # Removed mask for KV cache case
            )
            # print(f"Attention output y.shape={y.shape}")

        else:  # No KV cache branch - Use causal mask for training
            q, k = apply_rotary_embed(q, cos, sin), apply_rotary_embed(k, cos, sin)
            # print(f"q, k after rotary (no cache): q.shape={q.shape}, k.shape={k.shape}")
            q, k = rms_norm(q), rms_norm(k)
            # print(f"q, k after rms_norm (no cache): q.shape={q.shape}, k.shape={k.shape}")

            new_kv_cache = None

            if attn_mask is None:  # Keep mask for training (when kv_cache is None)
                attn_mask = jnp.tril(jnp.ones((T, T)), k=-1).astype(x.dtype)

            y = nnx.dot_product_attention(
                q,
                k,
                v,
                # mask=attn_mask,  # Use causal mask for training
            )
            # print(f"Attention output y.shape (no cache)={y.shape}")

        y = y.transpose(0, 2, 1, 3).reshape(x.shape)  # N T D
        # print(f"y reshaped, {y.shape = }")

        y = self.out_proj(y)
        # print(f"y out = {y.shape}")

        return (y, v1), new_kv_cache



class MLP(nnx.Module):
    def __init__(self, config=config):
        super().__init__()
        mlp_ratio = 2
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


# SwiGLU feedforward layer
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


# simple transformer decoder block
class DecoderBlock(nnx.Module):
    def __init__(self, config=config):
        self.attn = CausalAttention(config)
        self.mlp = MLP()
        self.rms_norm = nnx.RMSNorm(config.n_embd, rngs=rngs)

    def __call__(self, x, kv_cache=None, freq=None, v1=None):
        # print(f"DecoderBlock input x.shape: {x.shape}")  # ADDED PRINT
        (attn_out, v1), new_kv_cache = self.attn(
            self.rms_norm(x), kv_cache=kv_cache, freq=freq, v1=v1
        )
        x = x + attn_out
        x = x + self.mlp(self.rms_norm(x))

        return (x, v1), new_kv_cache


from functools import partial


# autoregressive transformer for image generation
class ART(nnx.Module):
    def __init__(self, config=config):
        self.layers = [DecoderBlock() for _ in range(config.depth)]
        self.transformer = nnx.Sequential(*self.layers)

        self.token_embed = nnx.Embed(
            config.vocab_size, config.n_embd, rngs=rngs, embedding_init=normal_init
        )
        self.linear_head = nnx.Linear(
            config.n_embd,
            config.vocab_size,
            rngs=rngs,
            kernel_init=zero_init,
            use_bias=False,
        )
        self.rotary = Rotary(config.n_embd // (2 * config.n_head))
        self.rms_norm = nnx.RMSNorm(config.n_embd, rngs=rngs)

    def __call__(self, token_idx, cond_labels):
        b, t = token_idx.shape
        c_tokens = cond_labels + 65536
        # print(f'{c_tokens.shape = }')
        targets = token_idx
        token_idx = token_idx[:, :-1]
        token_seq = jnp.concat([c_tokens[:, None], token_idx], axis=1)

        freq = self.rotary(token_idx, hw=(32, 32))

        x = self.token_embed(token_seq)
        x = x + x[:, 0:1, :]
        x = self.rms_norm(x)

        v1 = None
        for block in self.layers:
            x, v1 = block(x, freq=freq, v1=v1)[0]

        x = self.rms_norm(x)
        logits = self.linear_head(x).astype(config.dtype)

        one_hot_targets = jax.nn.one_hot(
            targets, num_classes=config.vocab_size
        ).reshape(-1, config.vocab_size)
        # print(f'{one_hot_targets.shape = } / {logits.reshape(-1, logits.shape[-1]).shape = }')

        loss = optax.softmax_cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            one_hot_targets,  # targets.reshape(-1).astype(jnp.int32)[:,None]
        ).mean()
        # print(loss)

        return logits, loss


    # @partial(nnx.jit, static_argnames=("max_tokens", "temp", "cfg_scale", "topk"))
    def generate(
        self, class_labels: Array, max_tokens=1024, temp=1.0, cfg_scale=4.0, topk=None
    ):
        b = class_labels.shape[0]
        # print(f"{class_labels.shape = }")

        class_labels = jnp.tile(class_labels, (2,))
        # print(f"repeated {class_labels.shape = }")

        class_labels = class_labels.at[b].set(1000)
        # class_labels[b:] = 1000
        x = (class_labels + 65536)[:, None]
        # print(f"x1 {x.shape}")

        kv_caches = [(0, 0)] * len(self.layers)
        x_init = self.token_embed(x)
        # print(f"{x_init.shape = }")

        freq = self.rotary(x_init, hw=(32, 32))
        # print(f"{freq[0].shape = } / {freq[1].shape = }")
        cos, sin = freq
        x_all = x

        for k in tqdm(range(max_tokens)):
            x_emb = self.token_embed(x_all[:, -1:])
            x_emb = x_emb + x_init
            # print(f"x_emb.shape before norm {x_emb.shape = }")  # ADDED PRINT 1

            x_emb = self.rms_norm(x_emb)
            cos_local = cos[:, k : k + 1, :, :]
            sin_local = sin[:, k : k + 1, :, :]

            freq_local = (cos_local, sin_local)
            v1 = None

            # print(f"x_emb.shape before block loop: {x_emb.shape}")  # ADDED PRINT 2
            for n, layer in enumerate(self.layers):
                # print(
                #     f"Block {n} input x_emb.shape: {x_emb.shape}"
                # )  # ADDED PRINT 3 - Moved inside loop
                (x_emb, v1), new_kv_cache = layer(
                    x_emb, kv_cache=kv_caches[n], freq=freq_local, v1=v1
                )
                kv_caches[n] = new_kv_cache

            x_emb = self.rms_norm(x_emb)
            logits = self.linear_head(x_emb)
            # print(f"post linear head {logits.shape = }")

            logits_cond = logits[:b, :]
            logits_uncond = logits[b:, :]
            logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
            logits = logits / temp
            # print(f"{logits.shape = }")

            if topk is not None:
                v, _ = jax.lax.top_k(logits, min(topk, logits.shape[-1]))
                logits = jnp.where(logits < v[:, [-1]], -jnp.inf, logits)

            probs = nnx.softmax(logits.squeeze(1), axis=-1)
            # print(f"{probs.shape = }")

            idx_next = jrand.categorical(randkey, probs, axis=1)
            # print(f"1 {idx_next.shape = }")

            idx_next = jnp.tile(idx_next, (2,))[:, None]
            # print(f"{idx_next.shape = } / {x_all.shape = }")

            x_all = jnp.concat([x_all, idx_next.reshape(x.shape)], axis=1)
            # print(f"{x_all[:, 1:].shape = }")

        return x_all[:, 1:]