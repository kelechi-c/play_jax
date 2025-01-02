import jax, math, optax
from jax import numpy as jnp, random as jrand, Array
from jax.sharding import NamedSharding as NS, Mesh, PartitionSpec as PS
from jax.experimental import mesh_utils
from flax import nnx

import numpy as np
import wandb, click, time, gc, os, pickle
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from functools import partial
from flax.training import train_state
import flax.traverse_util
from flax.serialization import to_state_dict, from_state_dict
from flax.core import freeze, unfreeze

jax.distributed.initialize()
# jax.config.update("jax_default_matmul_precision", "bfloat16")
JAX_TRACEBACK_FILTERING = "off"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
XLA_PYTHON_CLIENT_MEM_FRACTION = 0.20
JAX_DEFAULT_MATMUL_PRECISION = "bfloat16"


class config:
    depth = 12
    attn_heads = 8
    hidden_dim = 256
    mlp_dim = hidden_dim * 2
    vocab_size = 2048
    max_len = 128
    seed = 256
    key = jrand.key(seed)
    split = 10_000
    learn_rate = 1e-5


num_devices = jax.device_count()
devices = jax.devices()

print(f"found {num_devices} JAX device(s)")
for device in devices:
    print(f"{device} /")


mesh_devices = mesh_utils.create_device_mesh((num_devices,))
mesh = Mesh(mesh_devices, axis_names=("data",))
print(f"{mesh.shape = }")


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("loaded tokenizer")


class TextData(IterableDataset):
    def __init__(self, split=128):
        super().__init__()
        self.split = split
        self.dataset = load_dataset(
            "neuralwork/arxiver", split="train", streaming=True
        ).take(self.split)

    def __len__(self):
        return self.split

    def __iter__(self):
        for sample in self.dataset:
            tokens = tokenizer(
                sample["abstract"],
                truncation=True,
                return_tensors="np",
                padding="max_length",
                max_length=128,
            )

            token_ids = tokens["input_ids"]
            attn_mask = tokens["attention_mask"]

            token_ids, attn_mask = jnp.array(token_ids), jnp.array(attn_mask)

            yield {"input_ids": token_ids, "attention_mask": attn_mask}


xavier_init = nnx.initializers.xavier_uniform()
zero_init = nnx.initializers.constant(0.0)
normal_init = nnx.initializers.normal(0.02)
randkey = config.key


def sinusoidal_init(max_len=128, min_scale=1.0, max_scale=10000.0):

    def init(key, shape, dtype=np.float32):
        """Sinusoidal init."""
        del key, dtype
        d_feature = shape[-1]
        pe = np.zeros((max_len, d_feature), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
        div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
        pe[:, : d_feature // 2] = np.sin(position * div_term)
        pe[:, d_feature // 2 : 2 * (d_feature // 2)] = np.cos(position * div_term)
        pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
        return jnp.array(pe)

    return init


class MLP(nnx.Module):
    def __init__(self, embed_dim, rngs: nnx.Rngs):
        super().__init__()
        self.layernorm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.linear1 = nnx.Linear(
            embed_dim,
            2 * embed_dim,
            rngs=rngs,
            kernel_init=xavier_init,
            bias_init=zero_init,
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
        x = nnx.relu(self.linear1(x))
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
            decode=False,
            rngs=rngs,
            kernel_init=xavier_init,
            bias_init=zero_init,
        )
        self.ffn_layer = MLP(embed_dim, rngs=rngs)

    def __call__(self, x_token: Array) -> Array:
        # self.attention.init_cache(dtype=jnp.bfloat16, input_shape=x_token.shape)

        x = x_token + self.layernorm(self.attention(x_token))
        x = x + self.layernorm(self.ffn_layer(x))
        return x


# autoregressive transformer block, or just an SLM
class Transformer(nnx.Module):
    def __init__(
        self,
        n_layers,
        embed_dim,
        attn_heads,
        rngs: nnx.Rngs,
        vocab_size=config.vocab_size,
        config=config,
    ):
        super().__init__()
        embed_init = nnx.initializers.normal(stddev=1.0)

        self.wtoken_embed = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=embed_dim,
            embedding_init=nnx.initializers.normal(stddev=1.0),
            rngs=rngs,
        )
        print(f"embed state {nnx.state(self.wtoken_embed)}")

        self.posembed_shape = (1, config.max_len, embed_dim)

        pe_value = sinusoidal_init(max_len=config.max_len)(None, self.posembed_shape)
        self.pos_embed = nnx.Param(pe_value, rngs=rngs)
        jax.lax.stop_gradient(self.pos_embed.value)

        self.layernorm = nnx.LayerNorm(
            embed_dim, rngs=rngs, epsilon=1e-6, scale_init=nnx.initializers.ones_init()
        )

        self.decoder_layers = [
            DecoderBlock(rngs=rngs, embed_dim=embed_dim, attn_heads=attn_heads)
            for _ in range(n_layers)
        ]
        # self.decoder_layers = nnx.Sequential(*decoder_layers)
        self.linear_head = nnx.Linear(
            embed_dim,
            vocab_size,
            rngs=rngs,
            kernel_init=xavier_init,
            bias_init=zero_init,
        )

    def __call__(self, x_tokens: jax.Array) -> jax.Array:
        print(f"input {x_tokens}")
        token_embed = self.wtoken_embed(x_tokens)

        print(f"after token embed {token_embed}")

        x = token_embed + self.pos_embed.value
        print(f"after pos add {x}")
        for block in self.decoder_layers:
            x = block(x)

        print(f"after decoder blocks {x}")

        x = self.layernorm(x)
        print(f"before linear head {x}")

        x = self.linear_head(x)
        print(f"model out {x}")

        return x

    def generate(self, cond_token, max_outlen=128, temperature=1):
        pass


def _to_jax_array(x):
    if not isinstance(x, jax.Array):
        x = jnp.asarray(x)
    return x


# save model params in pickle file
def save_paramdict_pickle(model, filename="model.pkl"):
    params = nnx.state(model)
    params = jax.device_get(params)

    state_dict = to_state_dict(params)
    frozen_state_dict = freeze(state_dict)

    flat_state_dict = flax.traverse_util.flatten_dict(frozen_state_dict, sep=".")

    with open(filename, "wb") as f:
        pickle.dump(frozen_state_dict, f)

    return flat_state_dict


def load_paramdict_pickle(model, filename="model.pkl"):
    with open(filename, "rb") as modelfile:
        params = pickle.load(modelfile)

    params = unfreeze(params)
    params = flax.traverse_util.unflatten_dict(params, sep=".")
    params = from_state_dict(model, params)

    nnx.update(model, params)

    return model, params


def wandb_logger(key: str, project_name, run_name=None):  # wandb logger
    # initilaize wandb
    wandb.login(key=key)
    wandb.init(project=project_name, name=run_name or None)


class TrainState(train_state.TrainState):
    graphdef: nnx.GraphDef[Transformer]


def initialize_state(model: nnx.Module, mesh, optimizer):
    with mesh:
        graphdef, params = nnx.split(model, nnx.Param)
        state = TrainState.create(
            apply_fn=graphdef.apply, params=params, tx=optimizer, graphdef=graphdef
        )
        state = jax.tree.map(_to_jax_array, state)
        state_spec = nnx.get_partition_spec(state)
        state = jax.lax.with_sharding_constraint(state, state_spec)

    state_sharding = nnx.get_named_sharding(state, mesh)
    return state, state_sharding


def jax_collate(batch):
    tokens = jnp.stack([jnp.array(item["input_ids"]) for item in batch], axis=0)
    mask = jnp.stack(
        [jnp.array(item["attention_mask"]) for item in batch],
        axis=0,
    )

    return {
        "input_ids": tokens,
        "attention_mask": mask,
    }


# @jax.vmap
def modelpass(logits, tokens):
    # print(f"first, {logits.shape = } / {tokens.shape = }")

    output = logits.squeeze(1)[..., :-1, :]
    targets = tokens.squeeze(1)[..., 1:]

    output = output.reshape(-1, output.shape[-1])
    targets = jnp.expand_dims(targets.reshape(-1), axis=-1)  # jnp.expand_dims(axis=-1)

    # print(f'{targets.shape = } / {output.shape = }')
    output = output.astype(jnp.bfloat16)
    targets = targets.astype(jnp.int32)
    # print(f'outputs {output} \n\n targets {targets}')

    return output, targets


# compile multidevice versions of train/eval/predict step fn.
lm_model = Transformer(
    n_layers=12, embed_dim=768, attn_heads=12, rngs=nnx.Rngs(config.seed)
)
tx_optimizer = optax.adamw(learning_rate=1e-5)

state, state_sharding = initialize_state(lm_model, mesh, tx_optimizer)
data_sharding = NS(mesh, PS("data"))
rep_sharding = NS(mesh, PS())


@partial(
    jax.jit,
    in_shardings=(state_sharding, data_sharding),  # type: ignore
    out_shardings=(state_sharding, None),  # type: ignore
    # static_argnums=(2, 3),
    donate_argnums=0,
)  # type: ignore
def train_step(model_state, batch):

    def loss_func(params, batch):
        model = nnx.merge(model_state.graphdef, params)

        tokens = batch["input_ids"]
        logits = model(tokens)

        # logits = logits.astype(jnp.bfloat16)
        # tokens = tokens.astype(jnp.bfloat16)
        output, targets = modelpass(logits, tokens)
        # jax.debug.print(f'outputs {output}')
        loss = optax.softmax_cross_entropy(output, targets).mean()

        return loss

    grad_fn = jax.value_and_grad(loss_func)
    loss, grads = grad_fn(model_state.params, batch)
    new_state = model_state.apply_gradients(grads=grads)

    return new_state, loss


# jit_train_step = jax.jit(
#     train_step,
#     in_shardings=(
#         state_sharding,
#         data_sharding,
#         None,
#     ),  # type: ignore
#     out_shardings=(state_sharding),  # type: ignore
#     static_argnums=(2, 3),
#     donate_argnums=0,
# )


def batch_trainer(epochs, model_state, train_loader):
    train_loss = 0.0
    # model.train()

    # wandb_logger(key="", project_name="tinylm_jax")

    stime = time.time()

    batch = next(iter(train_loader))
    print("start overfitting.../")

    for epoch in tqdm(range(epochs)):
        model_state, train_loss = train_step(model_state, batch)
        print(f"epoch {epoch+1}/{epochs}, train loss => {train_loss}")
        # wandb.log({"train/loss": train_loss, "train/log_loss": math.log10(train_loss)})

        if epoch % 50 == 0:
            pass

        jax.clear_caches()
        # jax.clear_backends()
        gc.collect()

    etime = time.time() - stime
    print(
        f"overfit time for {epochs} epochs -> {etime/60:.4f} mins / {etime/60/60:.4f} hrs"
    )

    return model_state, train_loss


def trainer(epochs, model_state, train_loader):
    train_loss = 0.0
    # model.train()

    # wandb_logger(key="", project_name="tinylm_jax")
    stime = time.time()

    print("start training.../")

    for epoch in tqdm(range(epochs)):
        for step, batch in tqdm(enumerate(train_loader)):
            train_loss = train_step(model_state, batch)
            print(f"step {step}/{len(train_loader)}, train loss => {train_loss:.4f}")
            # wandb.log({"loss": train_loss, "log_loss": math.log10(train_loss)})

            if step % 200 == 0:
                pass

            jax.clear_caches()
            # jax.clear_backends()
            gc.collect()

        print(f"epoch {epoch+1}/{epochs}, train loss => {train_loss:.4f}")

    etime = time.time() - stime
    # wandb.log({"train/time_mins": etime / 60})
    print(f"train time for {epochs} epochs -> {etime/60/60:.4f} hrs")


@click.command()
@click.option("-r", "--run", default="single_batch")
@click.option("-e", "--epochs", default=30)
@click.option("-bs", "--batch_size", default=8)
def main(run, epochs, batch_size):

    textdata = TextData()
    train_loader = DataLoader(textdata, batch_size=batch_size, collate_fn=jax_collate)

    vs = next(iter(train_loader))
    print(f"sample token shape: {vs['input_ids'].shape}")

    rantoks = jrand.randint(randkey, (vs["input_ids"].shape), 1, 1000)

    testout = lm_model(rantoks)
    print(f"test output shape => {testout.shape}")

    n_params = sum([p.size for p in jax.tree.leaves(nnx.state(lm_model))])
    print(f"model parameters count: {n_params/1e6:.2f}M, ")

    # if run == "single_batch":
    #     model, loss = batch_trainer(epochs, state, train_loader=train_loader)
    #     wandb.finish()
    #     print(f"single batch training ended at loss: {loss:.4f}")

    # elif run == "train":
    #     print(f"you missed your train looop impl boy")


main()
