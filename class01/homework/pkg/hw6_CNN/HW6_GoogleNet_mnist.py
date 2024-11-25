import tensorflow as tf

from tensorflow.keras import datasets, layers, models, Model

from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout

# Helper libraires

import numpy as np

import matplotlib.pyplot as plt

import cv2




# MNIST / Fashion MNIST Dataset

fashion_mnist = tf.keras.datasets.fashion_mnist

(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()

print(f_image_train.shape)

print(f_label_train.shape)




f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0




f_image_train = tf.expand_dims(f_image_train, axis=3, name=None)

#f_label_train = tf.expand_dims(f_label_train, axis=3, name=None)

f_image_test  = tf.expand_dims(f_image_test, axis=3, name= None)

#f_label_test  = tf.expand_dims(f_label_test, axis=3, name= None)

f_image_train = tf.repeat(f_image_train, 3, axis=3)

#f_label_train = tf.repeat(f_label_train, 3, axis=3)

f_image_test  = tf.repeat(f_image_test, 3, axis=3, name=None)

#f_label_test  = tf.repeat(f_label_test, 3, axis=3, name=None)




f_image_val   = f_image_train[101:120,:,:,:]

f_image_train = f_image_train[:100,:,:,:]

f_label_val   = f_label_train[101:120]

f_label_train = f_label_train[:100]




#f_image_val   = f_image_train[-2000:,:,:,:]

#f_image_train = f_image_train[:-2000,:,:,:]

#f_label_val   = f_label_train[-2000:]

#f_label_train = f_label_train[:-2000]







#f_label_train = tf.keras.utils.to_categorical(f_label_train)

#f_label_test  = tf.keras.utils.to_categorical(f_label_test)

#f_label_val   = tf.keras.utils.to_categorical(f_label_val)

print(f_image_train.shape)

print(f_label_train.shape)

#print(f_label_train.shape)

#print(f_image_train.shape)




class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

'''

plt.figure(figsize=(28,28))

for i in range(10):

    plt.subplot(3,4,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(f_image_train[i])

    # The CIFAR labels happen to be arrays, 

    # which is why you need the extra index

    plt.xlabel(class_names[f_label_train[i]])

plt.show()

'''

''' arbituray model '''

'''

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(64, activation='relu'))

model.add(tf.keras.layers.Dense(10))

'''

''' googlenet : inception'''

def Inception(x, filters):

    '''[ref] https://mldlcvmjw.tistory.com/292'''

    '''https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D'''

    ''' 1x1 convolution '''

    path1 = Conv2D(filters=filters[0], kernel_size=(1,1), strides=1, padding = 'same', activation = 'relu')(x)




    ''' 1x1 convolution -> 3x3 convolution '''

    path2 = Conv2D(filters=filters[1][0], kernel_size=(1,1), strides=1, padding = 'same', activation = 'relu')(x)

    path2 = Conv2D(filters=filters[1][1], kernel_size=(3,3), strides=1, padding = 'same', activation = 'relu')(path2)




    ''' 1x1 convolution -> 5x5 convolution '''

    path3 = Conv2D(filters=filters[2][0], kernel_size=(1,1), strides=1, padding = 'same', activation = 'relu')(x)

    path3 = Conv2D(filters=filters[2][1], kernel_size=(5,5), strides=1, padding = 'same', activation = 'relu')(path3)




    ''' 3x3 Maxpooling -> 1x1 convolution '''

    path4 = MaxPooling2D(pool_size=(3,3), strides=1, padding = 'same')(x)

    path4 = Conv2D(filters=filters[3], kernel_size=(1,1), strides=1, padding = 'same', activation = 'relu')(path4)




    return Concatenate(axis=-1)([path1,path2,path3,path4])




''' googlenet'''

def Auxiliary_classifier(x, name=None):

    model=AveragePooling2D(pool_size=(5,5),strides=3,padding='valid')(x)    

    model=Conv2D(128,(1,1),1,padding='same',activation='relu')(model)    

    model=Flatten()(model)    

    model=Dense(256,activation='relu')(model)    

    model=Dropout(0.4)(model)    

    model=Dense(10,activation='softmax',name=name)(model)    

    return model







# stage 1 (conv, polling, normalize, conv, conv, normalize, pooling)

model_in = Input(shape=f_image_train.shape[1:])

model = tf.keras.layers.Resizing(\

    224, 224, interpolation="bilinear", crop_to_aspect_ratio=True,\

    input_shape=f_image_train.shape[1:], name="Resizing_Layer")(model_in)







#model = tf.keras.layers.Rescaling(1./255)(model)

model = Conv2D(filters=64, kernel_size=(7,7), strides=2, padding = 'same', activation = 'relu')(model)

model = MaxPooling2D(pool_size=(3,3), strides=2, padding = 'same')(model)

model = BatchNormalization()(model)




# stage 2

model = Conv2D(filters=64, kernel_size=(1,1), strides=1, padding='same', activation='relu')(model)

model = Conv2D(filters=192, kernel_size=(3,3), strides=1, padding='same', activation='relu')(model)

model = BatchNormalization()(model)

model = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(model)




# stage 3

model = Inception(model, [64, (96, 128), (16, 32), 32])

model = Inception(model, [128, (128, 192), (32, 96), 64])

model = MaxPooling2D(pool_size=(3,3), strides=2, padding = 'same')(model)




# stage 4

model = Inception(model, [192, (96, 208), (16, 48), 64]) #a

aux1  = Auxiliary_classifier(model, name = 'aux1')

model = Inception(model, [160, (112, 224), (24, 64), 64])#b

model = Inception(model, [128, (128, 256), (24, 64), 64])#c

model = Inception(model, [112, (144, 288), (32, 64), 64])#d

aux2  = Auxiliary_classifier(model, name = 'aux2')

model = Inception(model, [256, (160, 320), (32, 128), 128])#e

model = MaxPooling2D(pool_size=(3, 3), strides = 2, padding = 'same')(model)




# stage 5

model = Inception(model, [256, (160, 320), (32, 128), 128])

model = Inception(model, [384, (192, 384), (48, 128), 128])

model = AveragePooling2D(pool_size=(7,7), strides = 1, padding = 'valid')(model)




#stage 6

model = Flatten()(model)

model = Dropout(0.4)(model)

#model = Dense(480, activation='linear')

model = Dense(units=256, activation='linear')(model)

main_branch = Dense(units=10, activation='softmax', name='main')(model)

model_fin = Model(inputs=model_in, outputs=[main_branch, aux1, aux2])




model_fin.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy','kernel','bias'])




model_fin.fit(f_image_train, f_label_train, epochs=100, batch_size = 5)

model_fin.save('GoogleNet_mnist.keras')