from torch import bfloat16
import jax, math, optax, wandb
from jax import numpy as jnp, random as jrand
from flax import nnx
from flax.training import checkpoints
from einops import rearrange
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader, IterableDataset

class config:
    attn_heads = 8
    hidden_dim = 768
    mlp_dim = hidden_dim * 2
    vocab_size = 8192
    n_layers = 12
    seed = 256
    key = jrand.key(seed)
    split = 10000
    learn_rate = 1e-4

# confirm devices
print(f"JAX devices: {jax.local_devices()}")

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
                return_temsors="np",
                padding="max_length",
                max_length=128,
            )

            token_ids = tokens["input_ids"]
            attn_mask = tokens["attention_mask"]

            tokens, attn_mask = jnp.array(token_ids), jnp.array(attn_mask)

            yield {"input_ids": tokens, "attention_mask": attn_mask}


def jax_collate(batch):
    tokens = jnp.stack([jnp.array(item["input_ids"], dtype=jnp.bfloat16) for item in batch], axis=0)
    mask = jnp.stack(
        [jnp.array(item["attention_mask"], dtype=jnp.bfloat16) for item in batch],
        axis=0,
    )

    return {
        "input_ids": tokens,
        "attention_mask": mask,
    }


textdata = TextData()
train_loader = DataLoader(textdata, batch_size=16, collate_fn=jax_collate)

vs = next(iter(train_loader))
print(f"sample shape: {vs['input_ids'].shape}")


xavier_init = nnx.initializers.xavier_uniform()
zero_init = nnx.initializers.constant(0.0)
normal_init = nnx.initializers.normal(0.02)


class MLP(nnx.Module):
    def __init__(self, embed_dim, rngs: nnx.Rngs):
        super().__init__()
        self.layernorm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.linear1 = nnx.Linear(embed_dim, 2 * embed_dim, rngs=rngs)
        self.linear2 = nnx.Linear(2 * embed_dim, embed_dim, rngs=rngs)

    def __call__(self, x_input: jax.Array) -> jax.Array:
        x = self.layernorm(x_input)
        x = nnx.silu(self.linear1(x))
        x = self.linear2(x)

        return x


class DecoderBlock(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, embed_dim=embed_dim):
        super().__init__()
        self.layernorm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(num_heads=attn_heads, in_features=embed_dim, dtype=jnp.bfloat16, decode=True, rngs=rngs)
        self.ffn_layer = MLP(embed_dim, rngs=rngs)

    def __call__(self, x_token: jax.Array) -> jax.Array:
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
        self.layernorm = nnx.LayerNorm(embed_dim, rngs=rngs)
        decoder_layers = [DecoderBlock(rngs=rngs) for _ in range(n_layers)]
        self.decoder_layers = nnx.Sequential(*decoder_layers)
        self.lm_head = nnx.Linear(embed_dim, vocab_size, rngs=rngs)

    def __call__(self, x_tokens: jax.Array) -> jax.Array:
        b_size, token_len = x_tokens.shape
        pos = jnp.arange(0, token_len, device=x_tokens.device, dtype=jnp.int64)
        token_embed = self.wtoken_embed(x_tokens.astype(jnp.int64))
        pos_embed = self.pos_embed(pos)

        x = token_embed + pos_embed

        x = self.layernorm(self.decoder_layers(x_tokens))

        x = self.lm_head(x)
        output = nnx.softmax(x, axis=1)

        print(f"lm/softmax out shape => {output.shape}")

        return output

    def generate(self, cond_token, max_outlen=256, temperature=0.1):
        pass


slm_model = Transformer(n_layers, embed_dim, rngs=nnx.Rngs(3))
# nnx.display(slm_model)
s = slm_model(jnp.ones((1, 20, 768)))
print(s.shape)


graph, state = nnx.split(slm_model)
n_params = sum([p.size for p in jax.tree.leaves(state)])
print(f"number of parameters: {n_params/1e6:.3f}M")

optimizer = nnx.Optimizer(slm_model, optax.adamw(learning_rate=config.learn_rate))


def wandb_logger(key: str, project_name, run_name):  # wandb logger
    # initilaize wandb
    wandb.login(key=key)
    wandb.init(project=project_name, name=run_name)


def modelpass(logits, tokens):

    output = logits[..., :-1, :]
    targets = tokens[..., 1:].squeeze(1)

    output = output.reshape(-1, output.shape[-1])
    targets = targets.reshape(-1)

    output = output.astype(jnp.float32)
    targets = targets.astype(jnp.b)

    return output, targets


def loss_func(model, batch):
    tokens = batch["input_ids"]
    logits = model(tokens)

    logits = logits.astype(jnp.float32)
    tokens = tokens.astype(jnp.float32)

    output, targets = modelpass(logits, tokens)

    loss = optax.softmax_cross_entropy(output, targets).mean()

    return loss, logits


@nnx.jit
def train_step(model, optimizer, batch):
    gradfn = nnx.value_and_grad(loss_func, has_aux=True)
    (loss, logits), grads = gradfn(model, batch)
    optimizer.update(grads)

    return loss


def trainer(model, optimizer, train_loader):
    epochs = 1
    train_loss = 0.0
    model.train()
    # wandb_logger(
    #     key=None,
    #     model=model,
    #     project_name="transformer_playjax",
    #     run_name="tinygpt-1e-4-bs32-tpu",
    # )

    for epoch in tqdm(range(epochs)):
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

            train_loss = train_step(model, optimizer, batch)
            print(f"step {step}, loss-> {train_loss.item():.4f}")

            # wandb.log({"loss": train_loss.item()})

        print(f"epoch {epoch+1}, train loss => {train_loss}")


trainer(slm_model, optimizer, train_loader)
wandb.finish()
print("mini AR transformer training in JAX....done")
