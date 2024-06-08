# For getting the important disease labels
import tensorflow as tf
import cv2
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def get_image_files_and_labels(dir):
    for folder, _, filenames in tqdm(os.walk(dir)):
        for file in filenames:
            file_path = os.path.join(folder, file)

            label = folder.split('/')[4]
            yield (file_path, label)


def read_img(image_data):
    try:
        image = cv2.imread(image_data)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (200, 200))
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize the image
        return image
    
    except Exception as e:
        print(f'Error> {e}')
        


def load_images_from_directory(csv_file):
    image_folder = "/kaggle/input/plantvillage-dataset/color"
    
    data = pd.read_csv(csv_file).drop_duplicates(keep="first")

    paths = data["image_path"]
    captions = data["caption"]

    image_paths = [os.path.join(image_folder, file) for file in paths]

    return image_paths, captions


def load_image_features(csv):
    df = pd.read_csv(csv)

    image_feats = df["image"]
    labels = df["emotion_labels"]

    return image_feats, labels


def image_data_csv(images, labels):

    data_for_df = {"image": images, "labels": labels}

    feature_df = pd.DataFrame(data_for_df)
    feature_df.to_csv('plant_disease.csv', index=False)

    print("Image data csv creation complete")

    return feature_df
