import numpy as np
import cv2

### tuple
A = ("A", "B", )
B = ("C", )
C = A+B
print(C, type(C))


def print_number_square(n):
   
    num = 1

    
    for i in range(n):
        for j in range(n):
            print(num, end=' ')
            num += 1
        print()
n = int(input("Enter the size of the square (n): "))
print_number_square(n)


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
image_exp = np.expand_dims(image, 0)
print(image_exp.shape)
image_tr = np.transpose(image_exp, [0, 3, 1, 2])
print(image_tr.shape)



def custom_filter2D(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape


    pad_height = kernel_height // 2
    pad_width = kernel_width // 2


    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    output = np.zeros_like(image)


    for y in range(image_height):
        for x in range(image_width):
            sub_matrix = padded_image[y:y + kernel_height, x:x + kernel_width]
            output[y, x] = np.sum(sub_matrix * kernel)
    return output


img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)


kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])


output = custom_filter2D(img, kernel)


cv2.imshow('edge', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
