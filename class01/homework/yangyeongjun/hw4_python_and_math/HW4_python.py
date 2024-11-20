import cv2
import numpy as np

###01.튜플
my_tuple = ('A', 'B')
print (my_tuple)

new_value = 'C'
updated_tuple = my_tuple + (new_value,)
print(updated_tuple)

###02.숫자 사각형 만들기
n = int(input("정수 n을 입력하세요: "))



for i in range(0,n):
    print("")
    for j in range(0,n):
        print((j+1)+n*i, end='')
print("")

###03.reshape를 통한 1차원화
matrix = np.arange(1, n*n + 1).reshape(-1)
print(matrix)

###04.이미지 dims확장과 transpose
image = cv2.imread('lena.jpeg',cv2.COLOR_BGR2RGB)
print(image.shape)


#expand_dims
image_exp = np.expand_dims(image,0)
print(image_exp.shape)

#transpos
image_tr = np.transpose(image_exp,[0,3,1,2])
print(image_tr.shape)

###05.이미지 필터링과 가중치
img = cv2.imread('lena.jpeg', cv2.IMREAD_GRAYSCALE)
kernel = np.array([[1, 1, 1],[1, -8, 1], [1, 1, 1]])
print(kernel)
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('edge', output)
cv2.waitKey(0)