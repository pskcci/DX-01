import cv2
import numpy as np

# 이미지 읽기
img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("이미지 파일을 읽을 수 없습니다. 경로를 확인하세요.")
    exit()
# 커널 정의
kernel = np.array([[1, 1, 1],
                   [1, -8, 1],
                   [1, 1, 1]])

# 커널의 크기 (3x3)
kernel_height, kernel_width = kernel.shape

# 이미지 크기
img_height, img_width = img.shape

# 출력 이미지 초기화 (원본 이미지와 동일한 크기)
output = np.zeros_like(img)

# 이미지의 각 픽셀에 대해 합성 연산 수행
for i in range(1, img_height - 1):  # 가장자리를 제외한 내부 픽셀들
    for j in range(1, img_width - 1):
        # 현재 픽셀을 기준으로 커널 적용
        region = img[i-1:i+2, j-1:j+2]  # 3x3 영역 추출
        result = np.sum(region * kernel)  # 커널과 해당 영역의 곱셈 후 합산
        output[i, j] = np.clip(result, 0, 255)  # 결과 값이 0~255 사이에 오도록 제한

# 결과 출력
cv2.imshow('Edge Detection', output)
cv2.waitKey(0)
cv2.destroyAllWindows()