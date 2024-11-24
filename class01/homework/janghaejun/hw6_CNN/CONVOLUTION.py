import cv2
import numpy as np

#img = cv2.imread('img3.png', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('img3.png')
kernel_edge = np.array([[1, 1, 1],[1, -8, 1], [1, 1, 1]])
kernel_sharpen = np.array([[0, -1, 0],[-1, 5, -1], [0, -1, 0]])
kernel_box_blur = np.ones((3, 3), np.float32)/9
kernel_gaussian = np.array([[1, 4, 6, 4, 1],
                            [4, 16, 24, 16, 4],
                            [6, 24, 36, 24, 6],
                            [4, 16, 24, 16, 4],
                            [1, 4, 6, 4, 1]], dtype=np.float32)
kernel_gaussian /= 256 
print(kernel_edge)
print(kernel_sharpen)
print(kernel_box_blur)
print(kernel_gaussian)
output = cv2.filter2D(img, -1, kernel_edge)
output = cv2.filter2D(img, -1, kernel_sharpen)
output = cv2.filter2D(img, -1, kernel_box_blur)
output = cv2.filter2D(img, -1, kernel_gaussian)
cv2.imshow('kernel_edge', output)
cv2.imshow('kernel_sharpen', output)
cv2.imshow('kernel_box_blur', output)
cv2.imshow('kernel_gaussian', output)
cv2.waitKey(0)
