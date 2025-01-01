import jax, math, optax
from jax import numpy as jnp, random as jrand, Array
from jax.sharding import NamedSharding as NS, Mesh, PartitionSpec as PS
from jax.experimental import mesh_utils
from flax import nnx

from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader, IterableDataset

jax.config.update("jax_default_matmul_precision", "bfloat16")
JAX_TRACEBACK_FILTERING = "off"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
XLA_PYTHON_CLIENT_MEM_FRACTION = 0.20
JAX_DEFAULT_MATMUL_PRECISION = "bfloat16"


class config:
    depth = 12
    attn_heads = 8
    hidden_dim = 768
    mlp_dim = hidden_dim * 2
    vocab_size = 8192
    seed = 256
    key = jrand.key(seed)
    split = 10000
    learn_rate = 1e-4


hfdata = load_dataset("roneneldan/TinyStories", split="train", streaming=True).take(config.split)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


class TextData(IterableDataset):
    def __init__(self, dataset=hfdata):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return split

    def __iter__(self):
        for sample in self.dataset:
            tokens = tokenizer(
                sample["text"],
                truncation=True,
                return_tensors="np",
                padding="max_length",
                max_length=128,
            )

            token_ids = tokens["input_ids"]
            attn_mask = tokens["attention_mask"]

            tokens, attn_mask = jnp.array(token_ids), jnp.array(attn_mask)

            yield {"input_ids": tokens, "attention_mask": attn_mask}


xavier_init = nnx.initializers.xavier_uniform()
zero_init = nnx.initializers.constant(0.0)
normal_init = nnx.initializers.normal(0.02)
randkey = config.key


class MLP(nnx.Module):
    def __init__(self, embed_dim, rngs: nnx.Rngs):
        super().__init__()
        self.layernorm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.linear1 = nnx.Linear(
            embed_dim, 2 * embed_dim, rngs=rngs, 
            kernel_init=xavier_init, bias_init=zero_init
        )
        self.linear2 = nnx.Linear(
            2 * embed_dim,
            embed_dim,
            rngs=rngs,
            kernel_init=xavier_init,
            bias_init=zero_init,
        )

    def __call__(self, x_input: jax.Array) -> jax.Array:
        x = self.layernorm(x_input)
        x = nnx.silu(self.linear1(x))
        x = self.linear2(x)

        return x


class DecoderBlock(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, embed_dim=512, attn_heads=8):
        super().__init__()
        self.layernorm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=attn_heads, 
            in_features=embed_dim, 
            dtype=jnp.bfloat16,
            decode=True, rngs=rngs
        )
        self.ffn_layer = MLP(embed_dim, rngs=rngs)

    def __call__(self, x_token: Array) -> Array:
        x = x_token + self.layernorm(self.attention(x_token))
        x = x + self.layernorm(self.ffn_layer(x))
        return x

# autoregressive transformer block, or just an SLM
class Transformer(nnx.Module):
    def __init__(
        self, n_layers, embed_dim, rngs: nnx.Rngs, hidden_size=1024, vocab_size=32000
    ):
        super().__init__()
        embed_init = nnx.initializers.normal(0.02)
        
        self.wtoken_embed = nnx.Embed(
            vocab_size, embed_dim, rngs=rngs, embedding_init=embed_init
        )
        self.pos_embed = nnx.Embed(
            hidden_size, embed_dim, rngs=rngs, embedding_init=embed_init
        )

        self.layernorm = nnx.LayerNorm(embed_dim, rngs=rngs, epsilon=1e-6)

        self.decoder_layers = [DecoderBlock(rngs=rngs) for _ in range(n_layers)]
        # self.decoder_layers = nnx.Sequential(*decoder_layers)
        self.linear_head = nnx.Linear(
            embed_dim,
            vocab_size,
            rngs=rngs,
            kernel_init=xavier_init,
            bias_init=zero_init,
        )

    def __call__(self, x_tokens: jax.Array) -> jax.Array:
        b_size, token_len = x_tokens.shape
        pos = jnp.arange(0, token_len, device=x_tokens.device, dtype=jnp.int32)
        token_embed = self.wtoken_embed(x_tokens.astype(jnp.int32))
        pos_embed = self.pos_embed(pos)

        x = token_embed + pos_embed

        for block in self.decoder_layers:
            x = block(x)

        x = self.layernorm(x)
        x = self.linear_head(x)

        return x

    def generate(self, cond_token, max_outlen=128, temperature=1):
        pass