from jax import numpy as jnp, random as jrand, Array
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from functools import partial
from flax.training import train_state
import flax.traverse_util, builtins
from flax.serialization import to_state_dict, from_state_dict
from flax.core import freeze, unfreeze
import json
from safetensors import safe_open
from cosmos_tokenizer.image_lib import ImageTokenizer


class ImageTokenDataset(Dataset):
    def __init__(self, safetensor_path="./imagenet_di8x8.safetensors", debug=True):
        self.safetensor_path = safetensor_path
        self.split = 100_000
        
        metadata_path = safetensor_path.replace(".safetensors", "_metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
            self.total_samples = self.metadata["total_samples"]

        if debug:
            self.total_samples = 16

        with safe_open(safetensor_path, framework="flax") as f:
            self.indices = f.get_tensor("indices").to(jnp.uint16).astype(jnp.int64)[:self.split]
            self.labels = f.get_tensor("labels").astype(jnp.int64)[:self.split]

        if debug:
            self.indices = self.indices[:16]
            self.labels = self.labels[:16]

    def __len__(self):
        return int(self.total_samples)

    def __getitem__(self, idx):
        indices = self.indices[idx].reshape(-1)
        # replace randomly with 1000
        fill_cond = jrand.normal(randkey, (indices.shape)) < 0.05
        indices = jnp.where(fill_cond, indices, 1000)
        class_label = self.labels[idx]

        return {"input_ids": indices, "class_label": class_label}
