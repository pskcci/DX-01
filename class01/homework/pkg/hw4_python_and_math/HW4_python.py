import numpy as np
from PIL import Image 
import cv2

#-------- 실습 1번 -----------

# 기존 튜플
t = ('A', 'B',)

print(t)

# 새로운 요소 C 추가
t = t + ('C',)

# 결과 출력
print(t)


#-------- 실습 2번 -----------

# 정수 n 입력받기
n = int(input("정수를 입력하세요: "))
A = np.zeros((n,n))
# n x n 숫자 사각형 출력
for i in range(n):
    for j in range(n):
        A[i,j] = i*n+j  # 숫자 증가
    print(A)  # 줄 바꿈

#-------- 실습 3번 -----------

B = A.reshape(-1,)
print(B)


#-------- 실습 4번 -----------

# 임의의 이미지 파일 불러오기
image = cv2.imread('/home/intel/git-trianing/project-z/DX-01/class01/homework/pkg/images.jpg', cv2.COLOR_BAYER_BG2BGR)  # 이미지를 BGR 형식으로 읽기
print(image.shape)  # 이미지의 형태 출력

# 차원 확장
image_exp = np.expand_dims(image, 0)  # 이미지의 차원을 1 증가시킴
print(image_exp.shape)  # 확장된 이미지의 형태 출력

# 차원 순서 변경
image_tr = np.transpose(image_exp, [0,3,1,2])  # 이미지의 차원 순서를 변경하여 (배치, 채널, 높이, 너비) 형식으로 변환
print(image_tr.shape)  # 변환된 이미지의 형태 출력

#-------- 실습 5번 -----------


img = cv2.imread('/home/intel/git-trianing/project-z/DX-01/class01/homework/pkg/images.jpg', cv2.IMREAD_GRAYSCALE)  # 이미지를 그레이스케일로 읽기
kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])  # 엣지 검출을 위한 커널
print(kernel)

output = cv2.filter2D(img, -1, kernel)  # 필터 적용
cv2.imshow('edge', output)  # 결과 이미지 출력
cv2.waitKey(0)  # 키 입력 대기

