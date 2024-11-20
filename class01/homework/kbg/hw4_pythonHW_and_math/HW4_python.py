import math
import numpy as np
from PIL import Image
import cv2


def tupleTest():
    test_tuple = ("A", "B",)
    add_tuple = ("C",)
    merge_tuple = test_tuple + add_tuple
    print(merge_tuple)
    return merge_tuple

def intSquare():
    size = int(input("input intager number : "))
    int_square = [[0 for i in range(size)] for j in range(size)]
    num = 1
    for i in range(size):
        for j in range(size):
            int_square[i][j] = num
            num += 1
    for row in int_square:
        print(row)
    return int_square

def reshapeSquare(i_square):
    reshape_square = []
    for row in i_square:
        for num in row:
            reshape_square.append(num)
    print(reshape_square)
    return reshape_square

def imageTranspose(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_arr = np.array(image)
    #print(image_arr)
    print(f"Original shape: {image_arr.shape}")
    image_exp = np.expand_dims(image_arr, axis=0)
    #print(image_exp)
    print(f"Expanded shape: {image_exp.shape}")
    image_trans = np.transpose(image_exp, (0, 3, 2, 1))
    #print(image_trans)
    print(f"Transposed shape: {image_trans.shape}")
    return image_trans

def imageFilter(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    kernel = np.array([[1, 1, 1],[1, -8, 1], [1, 1, 1]])
    print(kernel)
    output = cv2.filter2D(image, -1, kernel)
    cv2.imshow('edge', output)
    cv2.waitKey(0)

# 1번 문제
tupleTest()
# 2번 문제
test = intSquare()
# 3번 문제
reshapeSquare(test)
# 4번 문제
imageTranspose("./Lena.png")
# 5번 문제
imageFilter("./Lena.png")