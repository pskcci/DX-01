#https://wikidocs.net/135874
#
import keras
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import GlobalAveragePooling2D
# Helper libraires
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import pickle

## tfds flower 102
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:40%]', 'train[40%:45%]', 'train[45%:50%]'],
    batch_size=32,
    with_info=True,
    as_supervised=True,
)
'''
https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub?hl=ko
'''
# get_label_name = metadata.features['label'].int2str
# print("get_label_name", get_label_name)

IMG_SIZE = 80
input_shape = (IMG_SIZE, IMG_SIZE, 3)
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip(),
    layers.RandomRotation(0.2),
])

resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMG_SIZE, IMG_SIZE),
    layers.Rescaling(1./255)
])

def prepare(ds, shuffle=False, augment=False):
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=AUTOTUNE)
    # Resize and rescale all datasets.
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y), 
               num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)
    # Batch all datasets.
    # ds = ds.batch(batch_size)
    # Use data augmentation only on the training set.

    ds.cache()
    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)

num_classes = metadata.features['label'].num_classes
get_label_name = metadata.features['label'].int2str

val_ds.cache()
val_ds.shuffle(buffer_size=1000)
test_ds.cache()
test_ds.shuffle(buffer_size=1000)

## Data augmentation
train_ds = prepare(train_ds, shuffle=True, augment=True)

image, label = next(iter(train_ds))
print(np.array(image).shape)
plt.figure(figsize=(5,5))
for i in range(10):
    plt.subplot(3,4,i+1)
    plt.imshow(image[i])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title(get_label_name(label[i]))
plt.show()

base_model = tf.keras.applications.MobileNetV2(
    input_shape=input_shape,
    include_top=False,
    weights='imagenet')

#preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

inputs = tf.keras.Input(shape=input_shape)
#x = preprocess_input(inputs)
# Mobilenet CNN model
x = base_model(inputs, training = True)
x = GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, output)
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_ds, 
                    validation_data=val_ds,
                    batch_size = 32,
                    #steps_per_epoch=1,
                    epochs=10)

model.save('transfer_learning.keras')
# with open('history', 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)