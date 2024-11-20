import numpy as np
import cv2
image = cv2.imread('/home/intel/다운로드/lena.png', cv2.COLOR_BAYER_BG2BGR)
print(image.shape)

image_exp = np.expand_dims(image, 0)
print(image_exp.shape)

image_tr = np.transpose(image_exp, [0,3,1,2])
print(image_tr.shape)