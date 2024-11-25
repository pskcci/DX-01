import numpy as np
import cv2

###prac1
tuple_data = ('A','B')
new_data = tuple_data +('C',)
print("update tuple: ", new_data)

###prac2&3
"""
import numpy as np

num = int(input("N = "))

for i in range(num**2):
    print(f"{i+1}", end =" ")
    if (i+1)%num == 0:
        print()

while True:
    a = int(input("line nun = "))
    if (num**2) % a == 0:
        line = a
        break
    else:
        print(f"{num**2}의 약수를 입력ㄱㄱ")

arr1 = np.arange(1,(num**2+1)).reshape(line,-1)
print(arr1)
"""
n = int(input())
A =np.zeros((n,n))
for i in range(n):
    for j in range(n):
        A[i,j] = i*n+j
print(A)

###prac4
### transpos
img = cv2.imread('lenna.jpeg', cv2.COLOR_BGR2RGB)
print(img.shape)

### expand_dims
img_exp = np.expand_dims(img, 0)
print(img_exp.shape)

### transpos
img_tr = np.transpose(img_exp, [0,3,1,2])
print(img_tr.shape)

###prac5
img = cv2.imread('lenna.jpeg', cv2.IMREAD_GRAYSCALE)
kernel = np.array([[1, 1, 1],[1, -8, 1], [1, 1, 1]])
kernel2 = np.array([[1, 4, 6, 4, 1],
                            [4, 16, 24, 16, 4],
                            [6, 24, 36, 24, 6],
                            [4, 16, 24, 16, 4],
                            [1, 4, 6, 4, 1]], dtype=np.float32)
kernel2 /= 256 
kernel3 = np.ones((3, 3), np.float32) / 9
print(kernel)
print(kernel2)
output1 = cv2.filter2D(img, -1, kernel)
output2 = cv2.filter2D(img, -1, kernel2)
output3 = cv2.filter2D(img, -1, kernel3)
cv2.imshow('win1', output1)
cv2.imshow('win2', output2)
cv2.imshow('win3', output2)
cv2.waitKey(0)