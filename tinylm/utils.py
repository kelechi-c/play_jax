import jax, wandb, pickle
from flax import nnx
from jax import Array, numpy as jnp
from flax.training.train_state import TrainState

import flax.traverse_util
from flax.serialization import to_state_dict, from_state_dict
from flax.core import freeze, unfreeze

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

    # print(type(params))
    params = unfreeze(params)
    # print(type(params))
    params = flax.traverse_util.unflatten_dict(params, sep=".")
    # print(type(params))
    params = from_state_dict(model, params)
    # print(type(params), type(model))

    nnx.update(model, params)

    return model, params


def wandb_logger(key: str, project_name, run_name=None):  # wandb logger
    # initilaize wandb
    wandb.login(key=key)
    wandb.init(project=project_name, name=run_name or None)


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

import jax
import jax.numpy as jnp
from jax import random
from flax import traverse_util


def generate_ar(
    model_state,
    initial_tokens: jax.Array,
    max_outlen: int,
    temperature: float = 1.0,
    rng = jrand.key(256),
    eos_token_id: int = tokenizer.eos_token_id,
):

    @jax.jit
    def predict_next_token(current_tokens, params, rng, top_k=2):
        # Add a batch dimension if it's not already there
        if current_tokens.ndim == 1:
            current_tokens = current_tokens[None, :]

        logits = model_state(current_tokens)
        # logits = model_state.apply_fn(
        #     {"params": params}, current_tokens
        # )  # Shape: (batch_size, seq_len, vocab_size)
        logits = logits[:, -1, :] / temperature  # Take logits for the last token

        # Stochastic sampling
        probabilities = jax.nn.softmax(logits, axis=-1)
        if top_k > 0:
            top_k_probs, top_k_indices = lax.top_k(probabilities, k=top_k)
            logprobs_top_k = jnp.log(top_k_probs)
            next_token = random.categorical(rng, logprobs_top_k)
            next_token = top_k_indices[jnp.arange(logits.shape[0]), next_token]
        else:
            next_token = random.categorical(rng, jnp.log(probabilities))

        return next_token[0]  # Remove batch dimension

    all_tokens = list(initial_tokens.tolist())

    def generate_next_token(carry, _):
        tokens, rng_carry = carry
        rng_sample, next_rng = random.split(rng_carry)
        next_token = predict_next_token(
            jnp.array(tokens), model_state.params, rng_sample
        )
        return (tokens + [next_token.item()], next_rng), next_token

    rng_iter = rng
    for _ in range(max_outlen - len(initial_tokens)):
        (all_tokens, rng_iter), next_token = generate_next_token(
            (all_tokens, rng_iter), None
        )
        if eos_token_id is not None and next_token == eos_token_id:
            break

    return jnp.array(all_tokens)

# Assuming you have your trained model_state and tokenizer
def ar_gen(model, tokenizer):
    # Example usage:
    initial_prompt = "The "
    initial_tokens = tokenizer.encode(initial_prompt, return_tensors="np")[0]  # Tokenize the prompt

    rng_generate = jrand.key(256)  # Separate RNG for generation
    generated_tokens = generate_ar(
        model_state,
        initial_tokens,
        max_outlen=128,
        temperature=1.0,
        rng=rng_generate,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(generated_tokens)
    print(f"Generated text: {generated_text}")


# a = nnx.MultiHeadAttention()