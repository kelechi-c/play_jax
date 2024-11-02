from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from katara import read_image
import jax, jax.numpy as jnp
import numpy as np
import cv2

class config:
    dataset_id = ""
    batch_size = 32
    split = 10000


def read_image(img, img_size: int = 32):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0

    return img


class ImageData(IterableDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return config.split

    def __iter__(self):
        for sample in self.dataset:
            image = sample["img"]
            image = read_image(image)

            image = jnp.array(image)
            label = jnp.array(sample["label"])

            yield image, label  # sample["label"]#label


def jax_collate(batch):
    images, labels = zip(*batch)
    batch = (jnp.array(images), jnp.array(labels))
    batch = jax.tree_util.tree_map(jnp.array, batch)
    return batch


traindata = ImageData()
train_loader = DataLoader(traindata, batch_size=32, collate_fn=jax_collate)

xc = next(iter(train_loader))

xc[0].shape, xc[1].shape
