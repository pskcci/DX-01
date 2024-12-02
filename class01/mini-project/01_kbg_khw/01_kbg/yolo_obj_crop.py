import cv2
import numpy as np
import torch
import os

def cameraOpen(model):
    cap = cv2.VideoCapture(0)
    y_onoff = False # YOLO 온오프 체커
    # 예외 발생 시 종료
    if not cap.isOpened():
        print("카메라를 열 수 없습니다. 다른 장치를 시도해 보세요.")
        return
    # 웹캠 항상 영상 재생
    while True:
        ret, frame = cap.read()
        if not ret:  # 이미지를 가져오지 못하면 break
            print("카메라에서 이미지를 가져올 수 없습니다. 카메라 연결을 확인하세요.")
            break
        if y_onoff:
            # 여기에 키보드 입력 처리 = 크롭 이미지 저장
            frame, detected, bx1, by1, bx2, by2 = yoloObjDetect(frame, model)
            # 프레임은 출력용
            # 4개 좌표는 바운딩박스 최상단 최하단 의미

        # 영상 출력
        if frame is not None:  # frame이 None이 아닌 경우만 표시
            cv2.imshow('Camera', frame)

        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 키로 종료
            break
        elif key == 32 and ret:  # Space 키
            # 여기에 키보드 입력 처리 = 크롭 이미지 저장
            if y_onoff:
                objCrop(frame, bx1, by1, bx2, by2)
        elif key == ord('y'):
            y_onoff = not y_onoff
            print(f"YOLO {'활성화' if y_onoff else '비활성화'}")

    # 카메라 릴리즈
    cap.release()
    # 창 닫기
    cv2.destroyAllWindows()

def yoloObjDetect(frame, model):
    """
    YOLOv5를 사용해 객체 탐지 및 가장 높은 정확도를 가진 객체 표시.
    Args:
        frame (np.ndarray): 웹캠 프레임
        model (torch.hub.load): YOLOv5 모델
    Returns:
        frame (np.ndarray): 객체 탐지 결과가 표시된 프레임
        detected (bool): 객체 탐지가 성공했는지 여부
    """
    # YOLO 모델로 프레임을 처리하여 결과를 얻음
    results = model(frame[..., ::-1])  # BGR에서 RGB로 변환
    
    # 탐지된 객체 정보 (데이터프레임)
    detections = results.pandas().xyxy[0]
    
    if len(detections) == 0:
        # 탐지된 객체가 없는 경우
        return frame, False, 0, 0, 0, 0

    # 가장 높은 정확도의 객체를 선택
    top_detection = detections.iloc[0]

    # 바운딩 박스 좌표 (x1, y1, x2, y2) 및 신뢰도 (confidence)
    x1, y1, x2, y2 = map(int, [top_detection['xmin'], top_detection['ymin'], top_detection['xmax'], top_detection['ymax']])
    confidence = top_detection['confidence']
    class_name = top_detection['name']
    
    # 프레임에 바운딩 박스 그리기
    color = (0, 255, 0)  # 초록색 (Bounding Box 색상)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 바운딩 박스

    # 클래스 이름과 신뢰도 표시
    label = f"{class_name} {confidence:.2f}"  # 클래스 이름과 신뢰도를 함께 표시
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, label, (x1, y1 - 10), font, 0.5, color, 2, cv2.LINE_AA)

    return frame, True, x1, y1, x2, y2

def objCrop(frame, x1, y1, x2, y2):
    """
    객체를 크롭하고 파일로 저장합니다.
    Args:
        frame (np.ndarray): 원본 이미지 (프레임)
        x1, y1, x2, y2 (int): 바운딩 박스 좌표 (xmin, ymin, xmax, ymax)
    """
    # 바운딩 박스 영역으로 이미지를 크롭
    crop_img = frame[y1:y2, x1:x2]

    # 'capture' 디렉토리로 저장 경로 변경
    save_dir = "capture"
    os.makedirs(save_dir, exist_ok=True)  # 디렉토리 생성
    
    # 파일 이름 생성
    save_path = os.path.join(save_dir, f"object_{x1}_{y1}_{x2}_{y2}.jpg")
    
    # 크롭된 이미지를 파일로 저장
    cv2.imwrite(save_path, crop_img)
    
    # 저장 완료 메시지 출력
    print(f"객체 이미지가 저장되었습니다: {save_path}")
    
    return crop_img  # 크롭된 이미지를 반환 (필요시 사용할 수 있도록)
    

def main():
    # YOLOv5 모델 로드
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 예시로 'yolov5s' 모델 로드
    # 카메라 실행
    cameraOpen(model)

if __name__ == "__main__":
    main()
