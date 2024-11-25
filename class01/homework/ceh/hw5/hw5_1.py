import numpy as np

# Sigmoid 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # 수정: exp(-x)로 수정

# Numerical Derivative 정의
def numerical_derivative(f, x):
    dx = 1e-4
    gradf = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])  # 수정: 유니코드 따옴표 오류 수정
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        
        # fx1 계산
        x[idx] = tmp_val + dx
        fx1 = f(x)
        
        # fx2 계산
        x[idx] = tmp_val - dx
        fx2 = f(x)
        
        gradf[idx] = (fx1 - fx2) / (2 * dx)

        # x 값을 원래 상태로 복원
        x[idx] = tmp_val
        it.iternext()

    return gradf

class LogicGate:
    def __init__(self, gate_name, xdata, tdata, learning_rate=0.01, threshold=0.5):
        self.name = gate_name
        self.__xdata = xdata.reshape(4, 2)
        self.__tdata = tdata.reshape(4, 1)  # 수정: tdata를 (4,1) 형태로 reshape
        self.__w = np.random.rand(2, 1)
        self.__b = np.random.rand(1)
        self.__learning_rate = learning_rate
        self.__threshold = threshold

    # 손실 함수 정의
    def __loss_func(self):
        delta = 1e-7
        z = np.dot(self.__xdata, self.__w) + self.__b
        y = sigmoid(z)
        return -np.sum(self.__tdata * np.log(y + delta) + (1 - self.__tdata) * np.log(1 - y + delta))  # 수정: 곱셈으로 변경

    # 오차 계산
    def err_val(self):
        delta = 1e-7
        z = np.dot(self.__xdata, self.__w) + self.__b
        y = sigmoid(z)
        return -np.sum(self.__tdata * np.log(y + delta) + (1 - self.__tdata) * np.log(1 - y + delta))  # 수정: 곱셈으로 변경

    # 학습 함수
    def train(self):
        f = lambda x: self.__loss_func()
        print("init error : ", self.err_val())

        for stp in range(20000):
            self.__w -= self.__learning_rate * numerical_derivative(f, self.__w)
            self.__b -= self.__learning_rate * numerical_derivative(f, self.__b)
            if stp % 2000 == 0:
                print("step : ", stp, "error : ", self.err_val())

    # 예측 함수
    def predict(self, input_data):
        input_data = input_data.reshape(1, -1)  # 입력을 2D 배열로 변환 (1, 2)
        z = np.dot(input_data, self.__w) + self.__b
        y = sigmoid(z)
        if y[0] > self.__threshold:
            result = 1
        else:
            result = 0
        return y, result


# AND 게이트
xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
tdata = np.array([[0], [0], [0], [1]])  # 수정: AND의 정답 데이터
AND = LogicGate("AND", xdata, tdata)
AND.train()
for in_data in xdata:
    sig_val, logic_val = AND.predict(in_data)
    print(in_data, " : ", logic_val)

# OR 게이트
xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
tdata = np.array([[0], [1], [1], [1]])  # 수정: OR의 정답 데이터
OR = LogicGate("OR", xdata, tdata)
OR.train()
for in_data in xdata:
    sig_val, logic_val = OR.predict(in_data)
    print(in_data, " : ", logic_val)

# NAND 게이트
xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
tdata = np.array([[1], [1], [1], [0]])  # NAND의 정답 데이터
NAND = LogicGate("NAND", xdata, tdata)
NAND.train()
for in_data in xdata:
    sig_val, logic_val = NAND.predict(in_data)
    print(in_data, " : ", logic_val)

# XOR 게이트 - 단층 신경망에서는 학습이 잘 안됩니다.
xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
tdata = np.array([[0], [1], [1], [0]])  # XOR의 정답 데이터
XOR = LogicGate("XOR", xdata, tdata)
XOR.train()
for in_data in xdata:
    sig_val, logic_val = XOR.predict(in_data)
    print(in_data, " : ", logic_val)