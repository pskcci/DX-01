import numpy as np


n = int(input("정수 n을 입력하세요: "))

matrix = np.arange(1, n*n + 1).reshape(-1)
print(matrix)