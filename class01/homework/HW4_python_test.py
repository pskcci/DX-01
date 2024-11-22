import numpy as np
import cv2
###tuple
A = ("A", "B", )
B = ("C",)
C = A+B
print(C, type(C))

###for loop
n = int(input())
A = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        A[i, j] = i*n+j
print(A)

### reshape
B= A.reshape(-1, )
print(B)

### transpos
image =  cv2.imread('/home/intel/다운로드/Lenna.png', cv2.COLOR_BGR2RGB)
print(image.shape)
### expand_dims
image_exp = np.expand_dims(image, 0)
print(image_exp.shape)

image_tr = np.transpose(image_exp, [0,3,1,2])
print(image_tr.shape)

### 이미지 읽기 (그레이스케일로)
image = cv2.imread('/home/intel/다운로드/Lenna.png', cv2.IMREAD_GRAYSCALE)

kernel = np.array([[1, 1, 1], 
                   [1, -8, 1], 
                   [1, 1, 1]])

kernel_height, kernel_width = kernel.shape
image_height, image_width = image.shape

output = np.zeros_like(image)

for i in range(1, image_height - 1):  
    for j in range(1, image_width - 1):  
        region = image[i-1:i+2, j-1:j+2]  
        result = np.sum(region * kernel)
        output[i, j] = np.clip(result, 0, 255)

cv2.imshow('edge', output)
cv2.waitKey(0)
cv2.destroyAllWindows()