import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Model load: MNIST / Fashion MNIST Dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
# or
#mnist = tf.keras.datasets.mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()
#(f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()
# normalized iamges
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#class_names = []
#for i in range(10):
   #class_names.append(i)

plt.figure(figsize=(10,10))
for i in range(10):
   plt.subplot(3,4,i+1)
   plt.xticks([])
   plt.yticks([])
   plt.grid(False)
   plt.imshow(f_image_train[i])
   plt.xlabel(class_names[f_label_train[i]])
plt.show()

# ANN
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
 )

model.fit(f_image_train, f_label_train, epochs=10, batch_size=100)
model.summary()
model.save('fashion_mnist.h5')
#model.save('font_mnist.h5')

model = tf.keras.models.load_model('./fashion_mnist.h5')
#model = tf.keras.models.load_model('./font_mnist.h5')
fashion_mnist = tf.keras.datasets.fashion_mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()

f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

num = 10
predict = model.predict(f_image_test[:num])
print(" * 실제: ", f_label_train[:num])
print(" * 예측 : ", np.argmax(predict, axis = 1))