import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

# BATCH_SIZE = 128
BATCH_SIZE = 32
model_filename = "mnist_cnn"

# set to True if we want to add insights to our data
insights = False

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
print(len(ds_train))
print(len(ds_test))
print(list(ds_train)[0][0])
list(ds_train)[0][0].shape # this is the shape we need our data in...

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

# normalise our pixel values so they fall between 0-1
ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

list(ds_train)[0][0].shape # this is the final shape we need!

# evaluation pipeline
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
#     tf.keras.layers.MaxPooling2D((2,2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
#     tf.keras.layers.Dense(10, activation = 'softmax')
# ]) # achieves 98.3% accuracy

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.Dense(10, activation = 'softmax')
]) # achieves 99% accuracy on val set
model.summary()

# LeNet-5 architecture for MNist
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(filters = 6, kernel_size = (5, 5), activation='tanh', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
#     tf.keras.layers.AvgPool2D((2,2), strides = 1),
#     tf.keras.layers.Conv2D(filters = 10, kernel_size = (5, 5), activation='relu', kernel_initializer='he_uniform', input_shape=(23, 23, 6)),
#     tf.keras.layers.AvgPool2D((2,2), strides = 2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(84, activation='tanh', kernel_initializer='he_uniform'),
#     tf.keras.layers.Dense(10, activation = 'softmax') # LeNet uses RBF as its activation, but I'm lazy
# ]) # achieves 98.9% accuracy on val set
# model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
model.fit(
        ds_train,
        epochs=10,
        validation_data=ds_test,
    )

import os
def save_model(filename, filepath, model):
    
    model_fname = filename
    my_wd = filepath

    model.save(os.path.join(my_wd, model_fname))
    print("Model saved")

# put into production our superior model!
save_model(model_filename, os.getcwd(), model)
