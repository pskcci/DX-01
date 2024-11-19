import numpy as np
from PIL import Image

# 1. 임의의 이미지 파일을 불러오기
image_path = "/home/intel/다운로드/images.jpeg"  # 이미지 파일 경로를 지정하세요.
image = Image.open(image_path)

# 이미지를 numpy 배열로 변환
image_np = np.array(image)

# 2. Numpy의 expand_dims를 사용해서 차원 확장 (Batch, Height, Width, Channel)
# 기존 이미지 shape: (Height, Width, Channel)
# expand_dims를 사용하여 배치 차원 추가
image_expanded = np.expand_dims(image_np, axis=0)

# 3. Numpy의 transpose를 사용해서 차원 순서 변경
# 기존 이미지 shape: (Batch, Height, Width, Channel)
# transpose로 (Batch, Channel, Height, Width)로 변경
image_transposed = np.transpose(image_expanded, (0, 3, 1, 2))

# 4. 결과 확인
print("원본 이미지 shape:", image_np.shape)
print("차원 확장 후 이미지 shape:", image_expanded.shape)
print("차원 순서 변경 후 이미지 shape:", image_transposed.shape)
