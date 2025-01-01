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
