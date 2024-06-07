import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from configs import config

train_data = tf.keras.utils.image_dataset_from_directory(
    config.data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(200, 200),
    batch_size=config.batch_size,
)

val_data = tf.keras.utils.image_dataset_from_directory(
    config.data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(200, 200),
    batch_size=config.batch_size,
)
