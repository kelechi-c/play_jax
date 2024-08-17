# For getting the important disease labels
import tensorflow as tf
import cv2
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def read_img(image_data):
    try:
        image = cv2.imread(image_data)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (200, 200))
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize the image
        return image

    except Exception as e:
        print(f"Error> {e}")
