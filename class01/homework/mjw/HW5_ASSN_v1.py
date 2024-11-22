import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

model = tf.keras.models.load_model('./f_mnist.keras')
# mnist = tf.keras.datasets.mnist
fashion_mnist = tf.keras.datasets.fashion_mnist

# (f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()
(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()

f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

num = 20

# train 이 맞는거 맞음 test는 오타임
predict = model.predict(f_image_train[:num])


print(f_label_train[:num])
print(" * Prediction, ", np.argmax(predict, axis = 1))