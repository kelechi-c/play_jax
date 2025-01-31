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
    def __init__(self, dim, base=100, h=32, w=32):
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


def rms_norm(x, dim=-1, target_rms=1.0, eps=1e-8):
    # return normalized_x
    rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=dim, keepdims=True) + eps)
    # 2. Normalize by dividing by RMS
    normalized_x = x / rms
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

class CausalAttention(nnx.Module):
    def __init__(self, config):
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
        # self.rms_norm = nnx.RMSNorm(dim, rngs=rngs)

    def __call__(
        self, x: jax.Array, kv_cache=None, freq: tuple = None, v1=None, pos=None
    ):
        N, T, D = x.shape  # T should always be 1 during generation
        # print(f'attn_input {x.shape = }')
        # Project inputs
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # Split into heads
        q, k, v = map(
            lambda x: rearrange(x, "b l (h d) -> b h l d", h=self.heads), (q, k, v)
        )

        cos, sin = freq

        # Process residual connection
        if v1 is not None:
            v = self.lamb1.value * v + self.lamb2.value * v1

        # Initialize new cache

        new_kv_cache = None

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # print(f"sample {q.shape = } / {cos.shape = }")

            q = apply_rotary_embed(q, cos, sin)
            k = apply_rotary_embed(k, cos, sin)

            q, k = rms_norm(q), rms_norm(k)

            if k_cache is None:  # Changed to k_cache is None
                cache_size = 1024
                k_cache = jnp.zeros((N, self.heads, cache_size, self.head_dim))
                v_cache = jnp.zeros((N, self.heads, cache_size, self.head_dim))

            k_cache = jax.lax.dynamic_update_slice(
                k_cache,
                k,
                (0, 0, pos, 0),
            )
            v_cache = jax.lax.dynamic_update_slice(
                v_cache,
                v,
                (0, 0, pos, 0),
            )

            new_kv_cache = (k_cache, v_cache)

            q = rearrange(q, "b h l d -> b l h d")
            k = rearrange(k, "b h l d -> b l h d")
            v = rearrange(v, "b h l d -> b l h d")

            y = nnx.dot_product_attention(q, k, v)
        else:
            # print(f"train {q.shape = } / {cos.shape = }")

            # Training path with full sequence
            q = apply_rotary_embed(q, cos, sin)
            k = apply_rotary_embed(k, cos, sin)
            q, k = rms_norm(q), rms_norm(k)

            q = rearrange(q, "b h l d -> b l h d")
            k = rearrange(k, "b h l d -> b l h d")
            v = rearrange(v, "b h l d -> b l h d")

            # print(f"train {q.shape = } / {k.shape = } / {v.shape = }")

            # Causal mask
            causal_mask = jnp.tril(jnp.ones((T, T)))[None, None, :, :]

            # mval = jnp.ones((T, T)) * -jnp.inf
            # attn_mask = jnp.triu(mval.astype(x.dtype), k=1)[None, :, :, None]
            # attn_mask = jnp.broadcast_to(attn_mask, (N, T, T, D))

            y = nnx.dot_product_attention(q, k, v, mask=causal_mask)

        # Merge heads and project
        # print(f'pre output = {y.shape}')
        y = rearrange(y, "b l h d -> b l (h d)")
        # print(f"arr = {y.shape}")

        y = self.out_proj(y)

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
        x = self.linear_2(x)
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
        # self.rms_norm = nnx.RMSNorm(config.n_embd, rngs=rngs)

    def __call__(self, x, kv_cache=None, freq=None, v1=None, pos=None):
        # print(f"DecoderBlock input x.shape: {x.shape}")  # ADDED PRINT
        (attn_out, v1), new_kv_cache = self.attn(
            rms_norm(x), kv_cache=kv_cache, freq=freq, v1=v1, pos=pos
        )
        x = x + attn_out
        x = x + self.mlp(rms_norm(x))

        return (x, v1), new_kv_cache

    def init_kv_cache(self, batch_size):
        # return {
        #     'key': jnp.zeros((batch_size, config.n_head, 0, self.attn.head_dim)),
        #     'value': jnp.zeros((batch_size, config.n_head, 0, self.attn.head_dim))
        # }
        return jnp.zeros((batch_size, config.n_head, 0, self.attn.head_dim)), jnp.zeros((batch_size, config.n_head, 0, self.attn.head_dim))


from functools import partial
from jax import lax
keyrng = jrand.PRNGKey(config.seed)
# nnx.split_rngs()

@jax.jit
def token_match_accuracy(generated_tokens: Array, actual_tokens: Array):
    # 1. Element-wise comparison: Check for token equality at each position
    token_matches = generated_tokens == actual_tokens
    # 2. Count the number of matching tokens
    num_correct_tokens = jnp.sum(token_matches)

    # 3. Calculate the total number of tokens being compared
    total_tokens = (
        generated_tokens.size
    )  # or actual_tokens.size, shapes should be the same

    # 4. Calculate accuracy: (number of correct tokens) / (total number of tokens)
    accuracy = num_correct_tokens / total_tokens

    # 5. Return accuracy as a Python float
    return accuracy


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
        # self.rms_norm = nnx.RMSNorm(config.n_embd, rngs=rngs)

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
        x = rms_norm(x)

        v1 = None
        for block in self.layers:
            x, v1 = block(x, freq=freq, v1=v1)[0]

        x = rms_norm(x)
        logits = self.linear_head(x).astype(config.dtype)

        gentokens = nnx.softmax(logits)
        gentokens = jrand.categorical(randkey, gentokens, axis=-1)
        # print(f'{gentokens.shape = } / {targets.shape = }')
        train_token_match = token_match_accuracy(gentokens, targets)

        # loss = optax.softmax_cross_entropy(output, one_hot_targets).mean()
        one_hot_targets = jax.nn.one_hot(
            targets, num_classes=config.vocab_size
        ).reshape(-1, config.vocab_size)
        # print(f'{one_hot_targets.shape = } / {logits.reshape(-1, logits.shape[-1]).shape = }')

        loss = optax.softmax_cross_entropy(
            logits.reshape(-1, logits.shape[-1]), one_hot_targets#targets.reshape(-1).astype(jnp.int32)[:,None]
        ).mean()
        # print(loss)
        # config.n_embd // config.n_head
        return logits, loss, train_token_match

    # @partial(nnx.ji, static_argnames=("max_tokens", "temp", "cfg_scale", "topk"))
    # @partial(
    #     nnx.jit,
    #     in_shardings=(rep_sharding, rep_sharding, data_sharding, data_sharding),
    #     out_shardings=(None, None),
    # )
    @nnx.jit
    def generate(
        self, class_labels: Array
    ):
        max_tokens=1024
        temp=1.0
        cfg_scale=4.0
        topk=None

        b = class_labels.shape[0]
        # print(f"{class_labels.shape = }")

        class_labels = jnp.tile(class_labels, (2,))
        # print(f"repeated {class_labels.shape = }")

        class_labels = class_labels.at[b].set(1000)
        # class_labels[b:] = 1000
        x = (class_labels + 65536)[:, None]
        # print(f"x1 {x.shape}")

        kv_caches = [None] * len(self.layers)  # [(0, 0)] * len(self.layers)
        x_init = self.token_embed(x)
        # print(f"{x_init.shape = }")
        x_out = jnp.zeros((32, 1024))

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
            print(f"{x_all[:, 1:].shape = }")

        x_out = x_all[:, 1:]
        return x_out
