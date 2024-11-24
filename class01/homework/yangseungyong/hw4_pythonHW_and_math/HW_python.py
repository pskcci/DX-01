import numpy as np
import cv2

### 1
AB = ('A', 'B')

C = 'C'

ABC = AB + (C,)

print(C,type(C)) 

### 2 & 3
print('정수 입력 n = ')
n = int(input())
A = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        A[i,j] = i*n+j

print(A)
B=A.reshape(-1,)
print(B)

### 4
image = cv2.imread('Lenna.png',cv2.COLOR_BGR2RGB)
print(image.shape)

### expand_dims
image_exp = np.expand_dims(image, 0)
print(image_exp.shape)


### transpos
image_tr = np.transpose(image_exp,[0,3,1,2])
print(image_tr.shape)


### 5
img = cv2.imread('Lenna.png',cv2.IMREAD_GRAYSCALE)
kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]])
print(kernel)

output = cv2.filter2D(img,-1,kernel)
cv2.imshow('edge',output)
cv2.waitKey(0)
