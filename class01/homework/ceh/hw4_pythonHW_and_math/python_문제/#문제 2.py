#문제 2

def print_number_square(n):
    # n x n 크기의 숫자 사각형 출력
    num = 1  # 숫자 1부터 시작
    for i in range(1, n+1):
        for j in range(1, n+1):
            # 순차적으로 숫자 출력
            print(num, end="\t")
            num += 1  # 출력 후 숫자 1 증가
        print()  # 한 줄 출력 후 줄바꿈

# 입력 받기
n = int(input("정수 n을 입력하세요: "))

# 함수 호출
print_number_square(n)
