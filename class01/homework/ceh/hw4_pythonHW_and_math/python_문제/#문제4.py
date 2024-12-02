
# 문제 4
#image = cv2.imread('/home/intel/사진/images.jpeg', cv2.color_RGR2RGB)
#print(image.shape)
###expand_dims
#image_exp = np.expand_dims(image, 0)
#print(image_exp.shape)
#### transpos
#image_tr = np.transpose(image_exp. [0,3,1,2])
#print(image_tr.shape)

#-----------------------------------------------

import cv2
import numpy as np

# 1. 이미지 파일 불러오기 
image = cv2.imread('/home/intel/사진/images.jpeg')

# 2. 이미지의 색상 공간을 BGR에서 RGB로 변환
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
print(image.shape)

# 3. Numpy의 expand_dims를 사용하여 차원 확장
image_exp = np.expand_dims(image, 0)  # (Batch, Height, Width, Channel)
print(image_exp.shape)

# 4. Numpy의 transpose를 사용하여 차원 순서 변경
# (Batch, Height, Width, Channel) -> (Batch, Channel, Height, Width)
image_tr = np.transpose(image_exp, (0, 3, 1, 2))  # (0, 3, 1, 2)로 수정
print(image_tr.shape)
