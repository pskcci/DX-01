import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

#Load model

model = tf.keras.models.load_model('./fashion_mnist.h5')
mnist = tf.keras.datasets.mnist
(f_image_train,f_label_train), (f_image_test,f_label_test) = mnist.load_data()

f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

num=10
predict = model.predict(f_image_train[:num])
print(f_label_train[:num])
print(" * prediction, ",np.argmax(predict,axis = 1))