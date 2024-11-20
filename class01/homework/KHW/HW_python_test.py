import numpy as np
import cv2

### tuple
A = ("A", "B", )
B = ("C", )
C = A+B
print(C, type(C))

### for loop
n = int(input())
A = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        A[i,j] = i*n+j
print(A)

### reshape
B = A.reshape(-1, )
print(B)

### transpos

image=cv2.imread('lenna.jpeg', cv2.COLOR_BGR2RGB)
print(type(image))
print(image.shape)
##### expand_dims
image_exp = np.expand_dims(image, 0)
print(image_exp.shape)
#### transpos
image_tr = np.transpose(image_exp, [0, 3, 1, 2])
print(image_tr.shape)