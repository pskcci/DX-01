# n을 입력받기
n = int(input("정수 n을 입력하세요: "))

# 숫자 출력 변수 초기화
num = 1

# n x n 크기의 사각형 출력
for i in range(n):
    for j in range(n):
        print(num, end=" ")
        num += 1
    print()  # 한 행을 출력하고 줄 바꿈
