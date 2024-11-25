import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

model = tf.keras.models.load_model('./text_mnist.h5')
mnist = tf.keras.datasets.mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0
class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


min=0
max=1000

predic = model.predict(f_image_test[min:max])
print(f_label_test[min:max])
print(" * Prediction, ", np.argmax(predic, axis = 1))

a = f_label_test[min:max]
b = np.argmax(predic, axis = 1)
c = a-b

accuracy = 0
for tmp in c:
    if (tmp == 0):
        accuracy +=1
    else:
        continue

print(accuracy/max)