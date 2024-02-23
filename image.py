# Import Necessary Libraries
import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 
import os
# tf.config.set_visible_devices([], 'GPU')

# Load the Dataset
dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
data_dir = tf.keras.utils.get_file('cats_and_dogs.zip', origin=dataset_url, extract=True) # Downloads a file from a URL if it not already in the cache.
                                                                                          # extract=True: True tries extracting the file as an Archive, like tar or zip.
# Preprocess the Dataset
## get the directory name of the extracted folder
data_dir = os.path.join(os.path.dirname(data_dir), 'cats_and_dogs_filtered')

## define the image size and batch size
img_size = (160, 160)
batch_size = 32

## load training and validation datasets
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split=0.2, 
                                                                    subset='training', seed=123, 
                                                                    image_size=img_size, shuffle=True, 
                                                                    batch_size=batch_size)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split=0.2, 
                                                                    subset='validation', seed=123, 
                                                                    image_size=img_size, shuffle=True, 
                                                                    batch_size=batch_size)

pretrained_model = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3),
                                               include_top=False,
                                               weights='imagenet')

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

inputs = tf.keras.Input(shape=(160, 160, 3))
x = preprocess_input(inputs)
x = pretrained_model(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(units=16, activation='relu')(x)
outputs = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


# model = tf.keras.models.Sequential([
#   pretrained_model,
#   tf.keras.layers.GlobalAveragePooling2D(),
#   tf.keras.layers.Dense(1)
# ])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset)