import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist

(f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()
# normalized iamges
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

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
model.fit(f_image_train, f_label_train, epochs=10, batch_size=1000)
model.summary()
model.save('mnist.keras')
####
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(x))


def numerical_derivative(f, x):
    dx = 1e-4
    gradf = np.zeros_like(x)
    it = np.nditer(x, flags = ['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float((tmp_val)+dx)
        fx1 = f(x)
        x[idx] = float((tmp_val)-dx)
        fx2 = f(x)
        gradf[idx] = (fx1-fx2)/(2*dx)
        x[idx] = tmp_val
        it.iternext()
    return gradf

class logicGate:
    def __init__(self, gate_name, xdata, tdata, learning_rate=0.01, threshold=0.5):
        self.name = gate_name

        self.__xdata=xdata.reshape(4,2)
        self.__tdata=tdata.reshape(4,1)

        self.__w=np.random.rand(2,1)
        self.__b=np.random.rand(1)

        self.__learning_rate = learning_rate
        self.__threshold = threshold

    def __loss_func(self):
        delta = 1e-7

        z = np.dot(self.__xdata, self.__w) + self.__b
        y = sigmoid(z)

        return -np.sum(self.__tdata*np.log(y+delta) + (1-self.__tdata)*np.log((1-y)+delta))


    def err_val(self):
        delta = 1e-7

        z = np.dot(self.__xdata, self.__w)+self.__b
        y = sigmoid(z)
        return -np.sum(self.__tdata + np.log(y+delta) + (1-self.__tdata)*np.log((1-y)+delta))

    def train(self):
        f = lambda x : self.__loss_func()
        print("init error : ", self.err_val())

        for stp in range(20000):
            self.__w -= self.__learning_rate * numerical_derivative(f, self.__w)
            self.__b -= self.__learning_rate * numerical_derivative(f, self.__b)
            if (stp % 2000 == 0):
                print("step : ", stp, "| error : ", self.err_val(), f)

    def predict(self, input_data):
        z = np.dot(input_data, self.__w) + self.__b
        y = sigmoid(z)

        if y[0] > self.__threshold:
            result = 1
        else :
            result = 0
        return y, result


xdata = np.array([[0,0], [0,1], [1,0], [1,1]])
tdata = np.array([[0,0,0,1]])

AND = logicGate("AND", xdata, tdata)
AND.train()
for in_data in xdata:
    (sig_val, logic_val) = AND.predict(in_data)
    print(in_data , " : ", logic_val)

xdata = np.array([[0,0], [0,1], [1,0], [1,1]])
tdata = np.array([[0,1,1,1]])

OR = logicGate("OR", xdata, tdata)
OR.train()
for in_data in xdata:
    (sig_val, logic_val) = OR.predict(in_data)
    print(in_data , " : ", logic_val)

xdata = np.array([[0,0], [0,1], [1,0], [1,1]])
tdata = np.array([[1,1,1,0]])

NAND = logicGate("NAND", xdata, tdata)
NAND.train()
for in_data in xdata:
    (sig_val, logic_val) = NAND.predict(in_data)
    print(in_data , " : ", logic_val)

xdata = np.array([[0,0], [0,1], [1,0], [1,1]])
tdata = np.array([[0,1,1,0]])

XOR = logicGate("XOR", xdata, tdata)
XOR.train()
for in_data in xdata:
    (sig_val, logic_val) = XOR.predict(in_data)
    print(in_data , " : ", logic_val)
####
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

model = tf.keras.models.load_model('./mnist.keras')
mnist = tf.keras.datasets.mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) =mnist.load_data()
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0
num = 10
predict = model.predict(f_image_train[:num])
print(f_label_train[:num])
print(" * Prediction, ", np.argmax(predict, axis = 1))
