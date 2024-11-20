n = int(input("정수 n을 입력하세요: "))



for i in range(0,n):
    print("")
    for j in range(0,n):
        print((j+1)+n*i, end='')
print("")