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