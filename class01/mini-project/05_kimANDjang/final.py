import cv2
import mediapipe as mp
import numpy as np
from openvino.runtime import Core

# 모델 파일 경로 설정
face_model_xml = "./model/face-detection-retail-0004.xml"
face_model_bin = "./model/face-detection-retail-0004.bin"
emotion_model_xml = "./model/emotions-recognition-retail-0003.xml"
emotion_model_bin = "./model/emotions-recognition-retail-0003.bin"

# 감정 레이블과 해당 배경 이미지 파일 경로
EMOTIONS = ["neutral", "happy", "sad", "surprise", "anger"]
BACKGROUND_IMAGES = {
    "neutral": "./backgrounds/neutral.webp",   # 중립 감정 배경 이미지
    "happy": "./backgrounds/happy.webp",       # 행복 감정 배경 이미지
    "sad": "./backgrounds/sad.jpeg",           # 슬픔 감정 배경 이미지
    "surprise": "./backgrounds/surprise.webp", # 놀람 감정 배경 이미지
    "anger": "./backgrounds/anger.jpg"        # 분노 감정 배경 이미지
}

# OpenVINO Runtime 초기화 및 모델 로드
core = Core()

# 얼굴 검출 모델 로드
face_model = core.read_model(model=face_model_xml, weights=face_model_bin)
face_compiled_model = core.compile_model(model=face_model, device_name="CPU")

# 감정 인식 모델 로드
emotion_model = core.read_model(model=emotion_model_xml, weights=emotion_model_bin)
emotion_compiled_model = core.compile_model(model=emotion_model, device_name="CPU")

# 모델 입력 및 출력 정보
face_input_layer = face_compiled_model.input(0)
face_output_layer = face_compiled_model.output(0)
emotion_input_layer = emotion_compiled_model.input(0)
emotion_output_layer = emotion_compiled_model.output(0)

############## PARAMETERS #######################################################

# Set these values to show/hide certain vectors of the estimation
draw_gaze = True
draw_full_axis = True
draw_headpose = False

# Gaze Score multiplier (Higher multiplier = Gaze affects headpose estimation more)
x_score_multiplier = 10
y_score_multiplier = 10

# Threshold of how close scores should be to average between frames
threshold = .3

# 각도 임계값 설정
pitch_threshold = 45 

#################################################################################

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=2,
    min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)

face_3d = np.array([
    [0.0, 0.0, 0.0],            # Nose tip
    [0.0, -330.0, -65.0],       # Chin
    [-225.0, 170.0, -135.0],    # Left eye left corner
    [225.0, 170.0, -135.0],     # Right eye right corner
    [-150.0, -150.0, -125.0],   # Left Mouth corner
    [150.0, -150.0, -125.0]     # Right mouth corner
    ], dtype=np.float64)

# Reposition left eye corner to be the origin
leye_3d = np.array(face_3d)
leye_3d[:,0] += 225
leye_3d[:,1] -= 175
leye_3d[:,2] += 135

# Reposition right eye corner to be the origin
reye_3d = np.array(face_3d)
reye_3d[:,0] -= 225
reye_3d[:,1] -= 175
reye_3d[:,2] += 135

# Gaze scores from the previous frame
last_lx, last_rx = 0, 0
last_ly, last_ry = 0, 0

# 시선 각도 계산 함수 (회전 벡터 -> 각도 변환)
def get_euler_angles(rvec):
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        x_angle = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
        z_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x_angle = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
        z_angle = 0
    return np.degrees(x_angle), np.degrees(y_angle), np.degrees(z_angle)

while cap.isOpened():
    success, frame= cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    # 이미지 전처리 (BGR -> RGB 변환)
    input_image = frame[..., ::-1]
    # input_image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    # 얼굴 검출 (OpenVINO face-detection 모델 사용)
    resized_frame = cv2.resize(input_image, (300, 300))
    input_blob = np.expand_dims(resized_frame.transpose(2, 0, 1), axis=0)

    # 얼굴 검출 추론 실행
    faces = face_compiled_model([input_blob])[face_output_layer]
    faces = faces.squeeze()

    # 기본 배경 이미지 (중립 상태로 설정)
    background_image = cv2.imread(BACKGROUND_IMAGES["neutral"])

    # 배경 이미지가 None인 경우를 처리
    if background_image is None:
        print("배경 이미지 로드 실패!")
        continue  # 배경 이미지 로드 실패 시, 다음 프레임으로 넘어감

    # 배경 이미지 크기 조정 (프레임 크기와 동일하게)
    background_image_resized = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))

    # 얼굴이 감지되면 감정 분석
    for face in faces:
        confidence = face[2]  # 얼굴 검출의 신뢰도
        if confidence > 0.5:  # 신뢰도가 50% 이상인 얼굴만 추출
            xmin, ymin, xmax, ymax = int(face[3] * frame.shape[1]), int(face[4] * frame.shape[0]), int(face[5] * frame.shape[1]), int(face[6] * frame.shape[0])

            # 얼굴 영역 추출
            face_region = frame[ymin:ymax, xmin:xmax]

            if face_region.size == 0:  # face_region이 비어있는지 확인
                print("얼굴 영역이 비어 있습니다.")
                continue

            resized_face = cv2.resize(face_region, (64, 64))
            input_blob_face = np.expand_dims(resized_face.transpose(2, 0, 1), axis=0)

            # 감정 인식 추론 실행
            emotion_results = emotion_compiled_model([input_blob_face])[emotion_output_layer]
            emotion_index = np.argmax(emotion_results)
            emotion = EMOTIONS[emotion_index]

            # 감정에 따른 배경 이미지 설정
            background_image = cv2.imread(BACKGROUND_IMAGES.get(emotion, BACKGROUND_IMAGES["neutral"]))

            # 배경 이미지가 None인 경우를 처리
            if background_image is None:
                print(f"감정 '{emotion}'에 대한 배경 이미지 로드 실패!")
                background_image = cv2.imread(BACKGROUND_IMAGES["neutral"])

            # 배경 이미지 크기 조정 (프레임 크기와 동일하게)
            background_image_resized = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))

            # 얼굴 영역 제외하고 배경만 표시
            mask = np.ones_like(frame, dtype=np.uint8)  # 마스크를 1로 초기화
            mask[ymin:ymax, xmin:xmax] = 0  # 얼굴 영역을 제외한 마스크

            # mask는 단일 채널 (1채널) 이미지여야 하므로 그대로 사용
            # 얼굴이 아닌 부분만 배경 이미지를 덮어씌우기 위해 bitwise_and 사용
            background_part = cv2.bitwise_and(background_image_resized, background_image_resized, mask=mask[:, :, 0])  # 1채널 mask 사용

            # 얼굴 영역은 원본 frame에서 가져오기
            frame[ymin:ymax, xmin:xmax] = face_region

            # 배경 이미지 덮어씌우기
            frame = cv2.add(background_part, frame)

            # 얼굴 영역에 감정 표시
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {emotion}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # convert img from BGR to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # To improve performance
    img.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(img)
    img.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    (img_h, img_w, img_c) = img.shape
    face_2d = []

    if not results.multi_face_landmarks:
      continue 

    for face_landmarks in results.multi_face_landmarks:
        face_2d = []
        for idx, lm in enumerate(face_landmarks.landmark):
            # Convert landmark x and y to pixel coordinates
            x, y = int(lm.x * img_w), int(lm.y * img_h)

            # Add the 2D coordinates to an array
            face_2d.append((x, y))
        
        # Get relevant landmarks for headpose estimation
        face_2d_head = np.array([
            face_2d[1],      # Nose
            face_2d[199],    # Chin
            face_2d[33],     # Left eye left corner
            face_2d[263],    # Right eye right corner
            face_2d[61],     # Left mouth corner
            face_2d[291]     # Right mouth corner
        ], dtype=np.float64)

        face_2d = np.asarray(face_2d)

        # Calculate left x gaze score
        if (face_2d[243,0] - face_2d[130,0]) != 0:
            lx_score = (face_2d[468,0] - face_2d[130,0]) / (face_2d[243,0] - face_2d[130,0])
            if abs(lx_score - last_lx) < threshold:
                lx_score = (lx_score + last_lx) / 2
            last_lx = lx_score

        # Calculate left y gaze score
        if (face_2d[23,1] - face_2d[27,1]) != 0:
            ly_score = (face_2d[468,1] - face_2d[27,1]) / (face_2d[23,1] - face_2d[27,1])
            if abs(ly_score - last_ly) < threshold:
                ly_score = (ly_score + last_ly) / 2
            last_ly = ly_score

        # Calculate right x gaze score
        if (face_2d[359,0] - face_2d[463,0]) != 0:
            rx_score = (face_2d[473,0] - face_2d[463,0]) / (face_2d[359,0] - face_2d[463,0])
            if abs(rx_score - last_rx) < threshold:
                rx_score = (rx_score + last_rx) / 2
            last_rx = rx_score

        # Calculate right y gaze score
        if (face_2d[253,1] - face_2d[257,1]) != 0:
            ry_score = (face_2d[473,1] - face_2d[257,1]) / (face_2d[253,1] - face_2d[257,1])
            if abs(ry_score - last_ry) < threshold:
                ry_score = (ry_score + last_ry) / 2
            last_ry = ry_score

        # The camera matrix
        focal_length = 1 * img_w
        cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                [0, focal_length, img_w / 2],
                                [0, 0, 1]])

        # Distortion coefficients 
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        _, l_rvec, l_tvec = cv2.solvePnP(leye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        _, r_rvec, r_tvec = cv2.solvePnP(reye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # Get rotational matrix from rotational vector
        l_rmat, _ = cv2.Rodrigues(l_rvec)
        r_rmat, _ = cv2.Rodrigues(r_rvec)

        # [0] changes pitch
        # [1] changes roll
        # [2] changes yaw
        # +1 changes ~45 degrees (pitch down, roll tilts left (counterclockwise), yaw spins left (counterclockwise))

        # Adjust headpose vector with gaze score
        l_gaze_rvec = np.array(l_rvec)
        l_gaze_rvec[2][0] -= (lx_score-.5) * x_score_multiplier
        l_gaze_rvec[0][0] += (ly_score-.5) * y_score_multiplier

        r_gaze_rvec = np.array(r_rvec)
        r_gaze_rvec[2][0] -= (rx_score-.5) * x_score_multiplier
        r_gaze_rvec[0][0] += (ry_score-.5) * y_score_multiplier

        # --- Projection ---
        l_corner = face_2d_head[2].astype(np.int32)
        axis = np.float32([[-100, 0, 0], [0, 100, 0], [0, 0, 300]]).reshape(-1, 3)
        l_axis, _ = cv2.projectPoints(axis, l_rvec, l_tvec, cam_matrix, dist_coeffs)
        l_gaze_axis, _ = cv2.projectPoints(axis, l_gaze_rvec, l_tvec, cam_matrix, dist_coeffs)

        # Get left eye corner as integer
        l_corner = face_2d_head[2].astype(np.int32)

        # Project axis of rotation for left eye
        axis = np.float32([[-100, 0, 0], [0, 100, 0], [0, 0, 300]]).reshape(-1, 3)
        l_axis, _ = cv2.projectPoints(axis, l_rvec, l_tvec, cam_matrix, dist_coeffs)
        l_gaze_axis, _ = cv2.projectPoints(axis, l_gaze_rvec, l_tvec, cam_matrix, dist_coeffs)
        
        # Calculate Euler angles for both eyes
        l_pitch, l_yaw, l_roll = get_euler_angles(l_gaze_rvec)

        # 각도가 임계값을 넘으면 경고 메시지 표시
        if abs(l_yaw) > pitch_threshold:
            cv2.putText(img, f'WARNING: PITCH {l_pitch:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Draw axis of rotation for left eye
        if draw_headpose:
            if draw_full_axis:
                cv2.line(img, l_corner, tuple(np.ravel(l_axis[0]).astype(np.int32)), (200,200,0), 3)
                cv2.line(img, l_corner, tuple(np.ravel(l_axis[1]).astype(np.int32)), (0,200,0), 3)
            cv2.line(img, l_corner, tuple(np.ravel(l_axis[2]).astype(np.int32)), (0,200,200), 3)

        if draw_gaze:
            if draw_full_axis:
                cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[0]).astype(np.int32)), (255,0,0), 3)
                cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[1]).astype(np.int32)), (0,255,0), 3)
            cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[2]).astype(np.int32)), (0,0,255), 3)

        # Get left eye corner as integer
        r_corner = face_2d_head[3].astype(np.int32)

        # Get left eye corner as integer
        r_axis, _ = cv2.projectPoints(axis, r_rvec, r_tvec, cam_matrix, dist_coeffs)
        r_gaze_axis, _ = cv2.projectPoints(axis, r_gaze_rvec, r_tvec, cam_matrix, dist_coeffs)

        # Draw axis of rotation for left eye
        if draw_headpose:
            if draw_full_axis:
                cv2.line(img, r_corner, tuple(np.ravel(r_axis[0]).astype(np.int32)), (200,200,0), 3)
                cv2.line(img, r_corner, tuple(np.ravel(r_axis[1]).astype(np.int32)), (0,200,0), 3)
            cv2.line(img, r_corner, tuple(np.ravel(r_axis[2]).astype(np.int32)), (0,200,200), 3)

        if draw_gaze:
            if draw_full_axis:
                cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[0]).astype(np.int32)), (255,0,0), 3)
                cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[1]).astype(np.int32)), (0,255,0), 3)
            cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[2]).astype(np.int32)), (0,0,255), 3)
            
    cv2.imshow('Head Pose Estimation', img)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
