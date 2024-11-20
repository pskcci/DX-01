import numpy as np
import cv2

image = cv2.imread('lena.jpeg',cv2.COLOR_BGR2RGB)
print(image.shape)


#expand_dims
image_exp = np.expand_dims(image,0)
print(image_exp.shape)

#transpos
image_tr = np.transpose(image_exp,[0,3,1,2])
print(image_tr.shape)
