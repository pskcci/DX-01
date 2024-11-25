import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

model = tf.keras.models.load_model('./fashion_mnist.h5')
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

num = 10
predict = model.predict(f_image_test[:num])
print(f_label_train[:num])
print(" * Prediction," , np.argmax(predict, axis=1))

mnist = tf.keras.datasets.mnist

model = Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(
    optimizer='adam',
    loss = 'sparse_Categorical_crossentropy',
    mertics=['accuracy'],
)
model.fit(image_train, label_train, epochs=10, batch_size=10)
model.summary()
model.save('fashion_mnist.h5')

