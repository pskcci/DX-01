import numpy as np

# n을 입력받기
n = int(input("정수 n을 입력하세요: "))

# 숫자 출력 변수 초기화
num = 1
matrix = []

# n x n 크기의 2차원 배열 만들기
for i in range(n):
    row = []
    for j in range(n):
        row.append(num)
        num += 1
    matrix.append(row)

# 2차원 배열을 numpy 배열로 변환
matrix_np = np.array(matrix)

# numpy 배열을 1차원 배열로 변환 (reshape)
reshaped_matrix = matrix_np.reshape(-1)

# 결과 출력
print("2차원 배열:")
print(matrix_np)
print("1차원 배열:")
print(reshaped_matrix)
