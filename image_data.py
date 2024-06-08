import tensorflow as tf
import math
from sklearn.preprocessing import LabelEncoder
from configs import config
import jax.numpy as jnp
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from util_funcs import *


file_paths, labels = zip(*get_image_files_and_labels(config.data_dir))

data_df = image_data_csv(file_paths, labels)


images = [img for img in data_df["images"]]

le = LabelEncoder()
enc_labels = le.fit_transform(data_df["labels"])

y_labels = list(data_df['labels'])

class ImageCaptionData(Dataset):
    def __init__(
        self,
        images,
        labels
    ):
        super().__init__()
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        image = read_img(img)
        
        print(image.shape)

        # disease label
        label = self.labels[idx]
        label = label.lower()
        print(label)

        return image, label


dataset = ImageCaptionData(images=images, labels=labels)

train_size = math.floor(len(dataset) * 0.9)
val_size = len(dataset) - train_size

train_data, valid_data = random_split(dataset, (train_size, val_size))

train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=config.batch_size, shuffle=False)