import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0
class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# fashion_mnist = tf.keras.datasets.fashion_mnist
# (f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()
# f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0
# class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
#              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#ANN
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.fit(f_image_train, f_label_train, epochs=5, batch_size=1000)
model.summary()
model.save('text_mnist.h5')


plt.figure(figsize=(10,10))
for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i])
    plt.xlabel(class_names[f_label_train[i]])
plt.show()
