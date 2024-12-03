import os
import time
import torch
import cv2
import numpy as np
import openvino as ov
from pathlib import Path

# 모델을 저장할 디렉토리 경로 설정
model_dir = "model/u2net"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# u2net.py 다운로드 및 임포트
if not os.path.exists("model/u2net.py"):
    import gdown
    gdown.download("https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/vision-background-removal/model/u2net.py", "model/u2net.py")

# u2net.py에서 U2NET 및 U2NETP 모델 클래스 임포트
from model.u2net import U2NET, U2NETP

# 모델 설정
u2net_lite = {
    'name': 'u2net_lite',
    'url': "https://drive.google.com/uc?id=1W8E4FHIlTVstfRkYmNOjbr0VDXTZm0jD",
    'model': U2NETP,
    'model_args': (),
}



# 사용할 모델을 선택하세요 (u2net 또는 u2net_lite)
u2net_model = u2net_lite  # u2net_lite로 변경하여 성능 향상

# 모델 다운로드 및 로드
model_path = os.path.join(model_dir, u2net_model['name'] + ".pth")
if not os.path.exists(model_path):
    import gdown
    gdown.download(u2net_model['url'], model_path)

# 네트워크 로드
net = u2net_model['model'](*u2net_model['model_args'])
net.eval()

# 모델 가중치 로드
net.load_state_dict(torch.load(model_path, map_location="cpu"))

# OpenVINO 모델로 변환
model_ir = ov.convert_model(net, example_input=torch.zeros((1, 3, 512, 512)), input=([1, 3, 512, 512]))

# 이미지 크기 설정 (512x512로 유지)
IMAGE_SIZE = (512, 512)  # 512x512로 변경

# 이미지 로드 및 전처리 함수
def process_frame(frame):
    # 웹캠에서 읽은 이미지를 전처리
    resized_image = cv2.resize(frame, IMAGE_SIZE)  # 크기 조정
    input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)

    input_mean = np.array([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
    input_scale = np.array([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)

    input_image = (input_image - input_mean) / input_scale

    return input_image

# 배경 이미지 경로 설정
BACKGROUND_FILE_1 = "/home/intel/openvino/mini_project/winter1.png"
BACKGROUND_FILE_2 = "/home/intel/openvino/mini_project/winter2.jpg"
BACKGROUND_FILE_3 = "/home/intel/openvino/mini_project/winter3.jpg"
BACKGROUND_FILE_4 = "/home/intel/openvino/mini_project/karina.png"

# 배경 이미지가 존재하는지 확인
if not os.path.exists(BACKGROUND_FILE_1):
    raise FileNotFoundError(f"첫 번째 배경 이미지 파일을 찾을 수 없습니다: {BACKGROUND_FILE_1}")
if not os.path.exists(BACKGROUND_FILE_2):
    raise FileNotFoundError(f"두 번째 배경 이미지 파일을 찾을 수 없습니다: {BACKGROUND_FILE_2}")
if not os.path.exists(BACKGROUND_FILE_3):
    raise FileNotFoundError(f"세 번째 배경 이미지 파일을 찾을 수 없습니다: {BACKGROUND_FILE_3}")
if not os.path.exists(BACKGROUND_FILE_4):
    raise FileNotFoundError(f"네 번째 배경 이미지 파일을 찾을 수 없습니다: {BACKGROUND_FILE_4}")

# OpenVINO 설정
core = ov.Core()
device = 'GPU'  # GPU로 변경
compiled_model_ir = core.compile_model(model=model_ir, device_name=device)

# 네트워크 결과 가져오기
input_layer_ir = compiled_model_ir.input(0)
output_layer_ir = compiled_model_ir.output(0)

# 프레임 스킵 설정
frame_skip = 2  # 2프레임마다 처리

# 비동기 요청을 위한 프레임 처리 함수
def process_frame_async(frame):
    input_image = process_frame(frame)  # 이미지 전처리
    return compiled_model_ir([input_image])[output_layer_ir]  # 모델 추론

# 웹캠에서 비디오 캡처
cap = cv2.VideoCapture(0)  # 웹캠 캡처 시작

# 웹캠이 열리지 않으면 종료
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# 배경 이미지 상태 추적 변수
current_background = 1

# 후처리 함수 (모폴로지 연산과 블러링 적용)
def post_process(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask

# 알파 블렌딩 함수 (배경과 전경을 자연스럽게 합성)
def alpha_blending(foreground, background, mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(foreground, 1, background, 1, 0)

frame_count = 0  # 프레임 카운터

while True:
    ret, frame = cap.read()  # 웹캠에서 한 프레임 읽기
    if not ret:
        print("Failed to capture image")
        break

    # 프레임 좌우 반전
    frame = cv2.flip(frame, 1)  # 좌우 반전 추가

    # 프레임 스킵
    if frame_count % frame_skip == 0:
        # 비동기적으로 추론 처리
        start_time = time.perf_counter()
        result = process_frame_async(frame)  # 비동기 처리
        end_time = time.perf_counter()

        # 후처리 및 화면 표시
        resized_result = np.rint(cv2.resize(np.squeeze(result), IMAGE_SIZE)).astype(np.uint8)
        smoothed_mask = post_process(resized_result)

        # 배경 제거 및 합성
        fg = frame.copy()

        # smoothed_mask를 원본 프레임 크기로 리사이즈
        smoothed_mask_resized = cv2.resize(smoothed_mask, (frame.shape[1], frame.shape[0]))

        # 리사이즈된 마스크를 사용하여 배경 제거
        fg[smoothed_mask_resized == 0] = 0

        # 배경 이미지 로드 및 크기 조정
        background_image = cv2.imread(BACKGROUND_FILE_1)
        background_image = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))

        background_image[smoothed_mask_resized == 1] = 0
        new_image = alpha_blending(fg, background_image, smoothed_mask_resized)

        # 화면에 표시
        cv2.imshow('Background Changed', new_image)  # 모델 출력 부분 주석 처리

    frame_count += 1  # 프레임 카운터 증가

    # 'ESC' 키로 종료, 'a' 키로 배경 전환
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('a'):
        current_background = 2 if current_background == 1 else 1

# 웹캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
