import cv2
import numpy as np

image = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
kernel = np.array([[1, 1, 1],[1, -8, 1],[1, 1, 1]])
print(kernel)
def filterTest(img, kern):
    output = cv2.filter2D(img, -1, kern)
    cv2.imshow('edge', output)
    cv2.waitKey(0)

k_identity = np.array([[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]])
k_ridge = np.array([[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]])
k_edgedetection = np.array([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]])
k_sharpen = np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]])
k_boxblur  = np.array([[1.0/9, 1.0/9, 1.0/9],
                       [1.0/9, 1.0/9, 1.0/9],
                       [1.0/9, 1.0/9, 1.0/9]])
k_gaussianblur = np.array([[1.0/16, 2.0/16, 1.0/16],
                           [2.0/16, 4.0/16, 2.0/16],
                           [1.0/16, 2.0/16, 1.0/16]])

filterTest(image, k_identity)
filterTest(image, k_ridge)
filterTest(image, k_edgedetection)
filterTest(image, k_sharpen)
filterTest(image, k_boxblur)
filterTest(image, k_gaussianblur)
