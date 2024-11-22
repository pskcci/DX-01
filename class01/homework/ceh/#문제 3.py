#문제 3
def print_number_square(n):
    # n x n 크기의 숫자 사각형 출력
    for i in range(1, n+1):
        for j in range(1, n+1):
            # 숫자 사각형에서 각 자리 숫자 출력
            print(i * j, end="\t")  # 숫자 출력 후 탭으로 구분
        print()  # 한 줄 출력 후 줄바꿈

def generate_number_square(n):
    # n x n 크기의 숫자 사각형을 1차원 리스트로 생성
    number_list = []
    
    for i in range(1, n+1):
        for j in range(1, n+1):
            number_list.append(i * j)  # 각 자리 숫자 계산 후 리스트에 추가
    
    return number_list  # 1차원 리스트로 반환

# 입력 받기 (하나만 받음)
n = int(input("정수 n을 입력하세요: "))

# 함수 호출
print_number_square(n)  # 숫자 사각형 출력

result = generate_number_square(n)  # 1차원 배열로 변환
print("1차원 리스트로 변환된 숫자 사각형:")
print(result)


