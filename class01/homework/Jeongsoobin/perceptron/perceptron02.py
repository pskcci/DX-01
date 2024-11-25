import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',]

plt.figure(figsize=(15,15))
for i in range(30):
    plt.subplot(6,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i])
    plt.xlabel(class_names[f_label_train[i]])
plt.show()