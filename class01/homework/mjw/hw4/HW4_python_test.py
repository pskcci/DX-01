import numpy as np
import cv2

### tuple
A =('A','B')
B = ('C',)
C = A+B
print(C, type(C))


###for loop
n = int(input("input:"))
A = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        A[i,j] = i*n+j
print(A)

### reshape  #(column, row)
B = A.reshape(-1,)
print(B)


### q4
# opencv color code - bgr / normal color - rgb
img = cv2.imread('./workdir/Lenna.png', cv2.COLOR_BGR2RGB)  
print("img.shape:",img.shape)
### expand_dims
img_exp = np.expand_dims(img, 0)
print("img_exp.shape:",img_exp.shape)
### transpos # swap order
img_tr = np.transpose(img_exp, [0,3,1,2])
print("img_tr.shape:",img_tr.shape)


##### q5
img1 = cv2.imread('./workdir/Lenna.png', cv2.IMREAD_GRAYSCALE)
kernel = np.array([[1, 1, 1],[1, -8, 1], [1, 1, 1]])
print(kernel)
output = cv2.filter2D(img1, -1, kernel)
cv2.imshow('edge', output)
cv2.waitKey(0)

