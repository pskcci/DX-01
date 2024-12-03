import os
import time
import torch
import cv2
import numpy as np
import openvino as ov
from pathlib import Path
import collections
from IPython import display
from numpy.lib.stride_tricks import as_strided
import requests

# Fetch `notebook_utils` module
r = requests.get(
    url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)
open("notebook_utils.py", "w").write(r.text)
import notebook_utils as utils

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
u2net_model = {  # u2net으로 변경하여 성능 향상
    'name': 'u2net',
    'url': "https://drive.google.com/uc?id=1W8E4FHIlTVstfRkYmNOjbr0VDXTZm0jD",
    'model': U2NET,  # U2NET 모델로 변경
    'model_args': (),
}

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
    resized_image = cv2.resize(frame, IMAGE_SIZE)  # 크기 조정
    input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)

    input_mean = np.array([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
    input_scale = np.array([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)

    input_image = (input_image - input_mean) / input_scale

    return input_image

# 배경 이미지 경로 설정
BACKGROUND_FILES = [
    "/home/intel/openvino/mini_project/winter1.png",
    "/home/intel/openvino/mini_project/winter2.png",
    "/home/intel/openvino/mini_project/karina.png",
    "/home/intel/openvino/mini_project/winter3.png",
    "/home/intel/openvino/mini_project/bg.png",
    "/home/intel/openvino/mini_project/bg2.png",
]

# 배경 이미지가 존재하는지 확인
for background_file in BACKGROUND_FILES:
    if not os.path.exists(background_file):
        raise FileNotFoundError(f"배경 이미지 파일을 찾을 수 없습니다: {background_file}")

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

# 배경 이미 상태 추적 변수
current_background = 0  # 현재 배경 인덱스

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

# Motion Detection 관련 코드
previous_right_hand_x = None  # 오른손의 이전 x 좌표를 저장할 변수 추가
previous_left_hand_x = None  # 왼쪽 손목의 이전 x 좌표
save_left_frame_time = None  # 왼쪽 손목 저장할 시간 기록 변수
save_right_frame_time = None  # 오른쪽 손목 저장할 시간 기록 변수

# OpenVINO 모델 로드 (motion.py의 코드)
base_model_dir = Path("model")
model_name = "human-pose-estimation-0001"
precision = "FP16-INT8"
model_path = base_model_dir / "intel" / model_name / precision / f"{model_name}.xml"

if not model_path.exists():
    model_url_dir = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/{model_name}/{precision}/"
    utils.download_file(model_url_dir + model_name + ".xml", model_path.name, model_path.parent)
    utils.download_file(
        model_url_dir + model_name + ".bin",
        model_path.with_suffix(".bin").name,
        model_path.parent,
    )

device = utils.device_widget()
core = ov.Core()
model = core.read_model(model_path)
compiled_model = core.compile_model(model=model, device_name=device.value)

# Get the input and output names of nodes.
input_layer = compiled_model.input(0)
output_layers = compiled_model.outputs
height, width = list(input_layer.shape)[2:]

# OpenPoseDecoder 클래스 정의 (motion.py의 코드)
class OpenPoseDecoder:
    BODY_PARTS_KPT_IDS = (
        (1, 2),
        (1, 5),
        (2, 3),
        (3, 4),
        (5, 6),
        (6, 7),
        (1, 8),
        (8, 9),
        (9, 10),
        (1, 11),
        (11, 12),
        (12, 13),
        (1, 0),
        (0, 14),
        (14, 16),
        (0, 15),
        (15, 17),
        (2, 16),
        (5, 17),
    )
    BODY_PARTS_PAF_IDS = (
        12,
        20,
        14,
        16,
        22,
        24,
        0,
        2,
        4,
        6,
        8,
        10,
        28,
        30,
        34,
        32,
        36,
        18,
        26,
    )

    def __init__(self, num_joints=18, skeleton=BODY_PARTS_KPT_IDS, paf_indices=BODY_PARTS_PAF_IDS, max_points=100, score_threshold=0.1, min_paf_alignment_score=0.05, delta=0.5):
        self.num_joints = num_joints
        self.skeleton = skeleton
        self.paf_indices = paf_indices
        self.max_points = max_points
        self.score_threshold = score_threshold
        self.min_paf_alignment_score = min_paf_alignment_score
        self.delta = delta

        self.points_per_limb = 10
        self.grid = np.arange(self.points_per_limb, dtype=np.float32).reshape(1, -1, 1)

    def __call__(self, heatmaps, nms_heatmaps, pafs):
        batch_size, _, h, w = heatmaps.shape
        assert batch_size == 1, "Batch size of 1 only supported"

        keypoints = self.extract_points(heatmaps, nms_heatmaps)
        pafs = np.transpose(pafs, (0, 2, 3, 1))

        if self.delta > 0:
            for kpts in keypoints:
                kpts[:, :2] += self.delta
                np.clip(kpts[:, 0], 0, w - 1, out=kpts[:, 0])
                np.clip(kpts[:, 1], 0, h - 1, out=kpts[:, 1])

        pose_entries, keypoints = self.group_keypoints(keypoints, pafs, pose_entry_size=self.num_joints + 2)
        poses, scores = self.convert_to_coco_format(pose_entries, keypoints)
        if len(poses) > 0:
            poses = np.asarray(poses, dtype=np.float32)
            poses = poses.reshape((poses.shape[0], -1, 3))
        else:
            poses = np.empty((0, 17, 3), dtype=np.float32)
            scores = np.empty(0, dtype=np.float32)

        return poses, scores

    def extract_points(self, heatmaps, nms_heatmaps):
        batch_size, channels_num, h, w = heatmaps.shape
        assert batch_size == 1, "Batch size of 1 only supported"
        assert channels_num >= self.num_joints

        xs, ys, scores = self.top_k(nms_heatmaps)
        masks = scores > self.score_threshold
        all_keypoints = []
        keypoint_id = 0
        for k in range(self.num_joints):
            mask = masks[0, k]
            x = xs[0, k][mask].ravel()
            y = ys[0, k][mask].ravel()
            score = scores[0, k][mask].ravel()
            n = len(x)
            if n == 0:
                all_keypoints.append(np.empty((0, 4), dtype=np.float32))
                continue
            x, y = self.refine(heatmaps[0, k], x, y)
            np.clip(x, 0, w - 1, out=x)
            np.clip(y, 0, h - 1, out=y)
            keypoints = np.empty((n, 4), dtype=np.float32)
            keypoints[:, 0] = x
            keypoints[:, 1] = y
            keypoints[:, 2] = score
            keypoints[:, 3] = np.arange(keypoint_id, keypoint_id + n)
            keypoint_id += n
            all_keypoints.append(keypoints)
        return all_keypoints

    def top_k(self, heatmaps):
        N, K, _, W = heatmaps.shape
        heatmaps = heatmaps.reshape(N, K, -1)
        ind = heatmaps.argpartition(-self.max_points, axis=2)[:, :, -self.max_points :]
        scores = np.take_along_axis(heatmaps, ind, axis=2)
        subind = np.argsort(-scores, axis=2)
        ind = np.take_along_axis(ind, subind, axis=2)
        scores = np.take_along_axis(scores, subind, axis=2)
        y, x = np.divmod(ind, W)
        return x, y, scores

    @staticmethod
    def refine(heatmap, x, y):
        h, w = heatmap.shape[-2:]
        valid = np.logical_and(np.logical_and(x > 0, x < w - 1), np.logical_and(y > 0, y < h - 1))
        xx = x[valid]
        yy = y[valid]
        dx = np.sign(heatmap[yy, xx + 1] - heatmap[yy, xx - 1], dtype=np.float32) * 0.25
        dy = np.sign(heatmap[yy + 1, xx] - heatmap[yy - 1, xx], dtype=np.float32) * 0.25
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        x[valid] += dx
        y[valid] += dy
        return x, y

    @staticmethod
    def is_disjoint(pose_a, pose_b):
        pose_a = pose_a[:-2]
        pose_b = pose_b[:-2]
        return np.all(np.logical_or.reduce((pose_a == pose_b, pose_a < 0, pose_b < 0)))

    def update_poses(self, kpt_a_id, kpt_b_id, all_keypoints, connections, pose_entries, pose_entry_size):
        for connection in connections:
            pose_a_idx = -1
            pose_b_idx = -1
            for j, pose in enumerate(pose_entries):
                if pose[kpt_a_id] == connection[0]:
                    pose_a_idx = j
                if pose[kpt_b_id] == connection[1]:
                    pose_b_idx = j
            if pose_a_idx < 0 and pose_b_idx < 0:
                pose_entry = np.full(pose_entry_size, -1, dtype=np.float32)
                pose_entry[kpt_a_id] = connection[0]
                pose_entry[kpt_b_id] = connection[1]
                pose_entry[-1] = 2
                pose_entry[-2] = np.sum(all_keypoints[connection[0:2], 2]) + connection[2]
                pose_entries.append(pose_entry)
            elif pose_a_idx >= 0 and pose_b_idx >= 0 and pose_a_idx != pose_b_idx:
                pose_a = pose_entries[pose_a_idx]
                pose_b = pose_entries[pose_b_idx]
                if self.is_disjoint(pose_a, pose_b):
                    pose_a += pose_b
                    pose_a[:-2] += 1
                    pose_a[-2] += connection[2]
                    del pose_entries[pose_b_idx]
            elif pose_a_idx >= 0 and pose_b_idx >= 0:
                pose_entries[pose_a_idx][-2] += connection[2]
            elif pose_a_idx >= 0:
                pose = pose_entries[pose_a_idx]
                if pose[kpt_b_id] < 0:
                    pose[-2] += all_keypoints[connection[1], 2]
                pose[kpt_b_id] = connection[1]
                pose[-2] += connection[2]
                pose[-1] += 1
            elif pose_b_idx >= 0:
                pose = pose_entries[pose_b_idx]
                if pose[kpt_a_id] < 0:
                    pose[-2] += all_keypoints[connection[0], 2]
                pose[kpt_a_id] = connection[0]
                pose[-2] += connection[2]
                pose[-1] += 1
        return pose_entries

    @staticmethod
    def connections_nms(a_idx, b_idx, affinity_scores):
        order = affinity_scores.argsort()[::-1]
        affinity_scores = affinity_scores[order]
        a_idx = a_idx[order]
        b_idx = b_idx[order]
        idx = []
        has_kpt_a = set()
        has_kpt_b = set()
        for t, (i, j) in enumerate(zip(a_idx, b_idx)):
            if i not in has_kpt_a and j not in has_kpt_b:
                idx.append(t)
                has_kpt_a.add(i)
                has_kpt_b.add(j)
        idx = np.asarray(idx, dtype=np.int32)
        return a_idx[idx], b_idx[idx], affinity_scores[idx]

    def group_keypoints(self, all_keypoints_by_type, pafs, pose_entry_size=20):
        all_keypoints = np.concatenate(all_keypoints_by_type, axis=0)
        pose_entries = []
        for part_id, paf_channel in enumerate(self.paf_indices):
            kpt_a_id, kpt_b_id = self.skeleton[part_id]
            kpts_a = all_keypoints_by_type[kpt_a_id]
            kpts_b = all_keypoints_by_type[kpt_b_id]
            n = len(kpts_a)
            m = len(kpts_b)
            if n == 0 or m == 0:
                continue

            a = kpts_a[:, :2]
            a = np.broadcast_to(a[None], (m, n, 2))
            b = kpts_b[:, :2]
            vec_raw = (b[:, None, :] - a).reshape(-1, 1, 2)

            steps = 1 / (self.points_per_limb - 1) * vec_raw
            points = steps * self.grid + a.reshape(-1, 1, 2)
            points = points.round().astype(dtype=np.int32)
            x = points[..., 0].ravel()
            y = points[..., 1].ravel()

            part_pafs = pafs[0, :, :, paf_channel : paf_channel + 2]
            field = part_pafs[y, x].reshape(-1, self.points_per_limb, 2)
            vec_norm = np.linalg.norm(vec_raw, ord=2, axis=-1, keepdims=True)
            vec = vec_raw / (vec_norm + 1e-6)
            affinity_scores = (field * vec).sum(-1).reshape(-1, self.points_per_limb)
            valid_affinity_scores = affinity_scores > self.min_paf_alignment_score
            valid_num = valid_affinity_scores.sum(1)
            affinity_scores = (affinity_scores * valid_affinity_scores).sum(1) / (valid_num + 1e-6)
            success_ratio = valid_num / self.points_per_limb

            valid_limbs = np.where(np.logical_and(affinity_scores > 0, success_ratio > 0.8))[0]
            if len(valid_limbs) == 0:
                continue
            b_idx, a_idx = np.divmod(valid_limbs, n)
            affinity_scores = affinity_scores[valid_limbs]

            a_idx, b_idx, affinity_scores = self.connections_nms(a_idx, b_idx, affinity_scores)
            connections = list(
                zip(
                    kpts_a[a_idx, 3].astype(np.int32),
                    kpts_b[b_idx, 3].astype(np.int32),
                    affinity_scores,
                )
            )
            if len(connections) == 0:
                continue

            pose_entries = self.update_poses(
                kpt_a_id,
                kpt_b_id,
                all_keypoints,
                connections,
                pose_entries,
                pose_entry_size,
            )

        pose_entries = np.asarray(pose_entries, dtype=np.float32).reshape(-1, pose_entry_size)
        pose_entries = pose_entries[pose_entries[:, -1] >= 3]
        return pose_entries, all_keypoints

    @staticmethod
    def convert_to_coco_format(pose_entries, all_keypoints):
        num_joints = 17
        coco_keypoints = []
        scores = []
        for pose in pose_entries:
            if len(pose) == 0:
                continue
            keypoints = np.zeros(num_joints * 3)
            reorder_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
            person_score = pose[-2]
            for keypoint_id, target_id in zip(pose[:-2], reorder_map):
                if target_id < 0:
                    continue
                cx, cy, score = 0, 0, 0  # keypoint not found
                if keypoint_id != -1:
                    cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                keypoints[target_id * 3 + 0] = cx
                keypoints[target_id * 3 + 1] = cy
                keypoints[target_id * 3 + 2] = score
            coco_keypoints.append(keypoints)
            scores.append(person_score * max(0, (pose[-1] - 1)))  # -1 for 'neck'
        return np.asarray(coco_keypoints), np.asarray(scores)

decoder = OpenPoseDecoder()

# 2D pooling in numpy
def pool2d(A, kernel_size, stride, padding, pool_mode="max"):
    A = np.pad(A, padding, mode="constant")
    output_shape = (
        (A.shape[0] - kernel_size) // stride + 1,
        (A.shape[1] - kernel_size) // stride + 1,
    )
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(
        A,
        shape=output_shape + kernel_size,
        strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides,
    )
    A_w = A_w.reshape(-1, *kernel_size)

    if pool_mode == "max":
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == "avg":
        return A_w.mean(axis=(1, 2)).reshape(output_shape)

# non maximum suppression
def heatmap_nms(heatmaps, pooled_heatmaps):
    return heatmaps * (heatmaps == pooled_heatmaps)

# Get poses from results.
def process_results(img, pafs, heatmaps):
    pooled_heatmaps = np.array([[pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]])
    nms_heatmaps = heatmap_nms(heatmaps, pooled_heatmaps)

    poses, scores = decoder(heatmaps, nms_heatmaps, pafs)
    output_shape = list(compiled_model.output(index=0).partial_shape)
    output_scale = (
        img.shape[1] / output_shape[3].get_length(),
        img.shape[0] / output_shape[2].get_length(),
    )
    poses[:, :, :2] *= output_scale
    return poses, scores

colors = (
    (255, 0, 0),
    (255, 0, 255),
    (170, 0, 255),
    (255, 0, 85),
    (255, 0, 170),
    (85, 255, 0),
    (255, 170, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 255, 85),
    (170, 255, 0),
    (0, 85, 255),
    (0, 255, 170),
    (0, 0, 255),
    (0, 255, 255),
    (85, 0, 255),
    (0, 170, 255),
)

default_skeleton = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)

def draw_poses(img, poses, point_score_threshold, skeleton=default_skeleton):
    if poses.size == 0:
        return img

    for pose in poses:
        points = pose[:, :2].astype(np.int32)
        points_scores = pose[:, 2]
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img, tuple(p), 1, colors[i], 2)
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                cv2.line(
                    img,
                    tuple(points[i]),
                    tuple(points[j]),
                    color=colors[j],
                    thickness=4,
                )
    return img

# Main processing function to run pose estimation.
def run_pose_estimation(frame):
    input_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    input_img = input_img.transpose((2, 0, 1))[np.newaxis, ...]

    results = compiled_model([input_img])
    pafs = results[compiled_model.output("Mconv7_stage2_L1")]
    heatmaps = results[compiled_model.output("Mconv7_stage2_L2")]
    poses, scores = process_results(frame, pafs, heatmaps)

    return poses

previous_left_hand_y = None  # 이전 왼손 y 좌표 초기화
save_sequence = 0  # 저장 순번 초기화 (루프 외부에서 초기화)

while True:
    ret, frame = cap.read()  # 웹캠에서 한 프레임 읽기
    if not ret:
        print("Failed to capture image")
        break

    # 좌우 반전
    frame = cv2.flip(frame, 1)  # 좌우 반전 추가
    
    # 포즈 추정 실행
    poses = run_pose_estimation(frame)

    # 오른손의 위치 추적
    if poses.size > 0:
        right_shoulder_x = poses[0][2][0]  # 오른쪽 어깨의 x 좌표 (keypoint ID 2)
        right_wrist_x = poses[0][5][0]  # 오른쪽 손목의 x 좌표 (keypoint ID 5)

        if previous_right_hand_x is not None:
            distance = abs(right_wrist_x - right_shoulder_x)
            proximity_threshold = 45
            max_proximity_threshold = 200

            if distance < proximity_threshold or distance > max_proximity_threshold:
                continue  # 모션 인식하지 않음

            movement_threshold = 8
            max_movement_threshold = 13

            # 오른손 모션이 동작 중인 경우 왼손 모션 인식 방지
            if right_wrist_x != previous_right_hand_x:  # 오른손 모션이 감지되면
                previous_left_hand_y = None  # 왼손 모션 초기화

            if (abs(previous_right_hand_x - right_wrist_x) > movement_threshold and
                abs(previous_right_hand_x - right_wrist_x) < max_movement_threshold):
                if right_wrist_x > previous_right_hand_x and right_wrist_x > right_shoulder_x:
                    print("change")  # 오른손이 오른쪽으로 이동하면 배경 변경
                    current_background = (current_background + 1) % len(BACKGROUND_FILES)  # 배경 변경

        previous_right_hand_x = right_wrist_x  # 이전 오른손 x 좌표 업데이트

    # 왼손의 위치 추적
    if poses.size > 0:
        left_elbow_y = poses[0][3][1]  # 왼쪽 팔꿈치의 y 좌표 (keypoint ID 3)
        left_wrist_y = poses[0][4][1]  # 왼쪽 손목의 y 좌표 (keypoint ID 4)

        if previous_left_hand_y is not None:
            # 왼쪽 손목과 팔꿈치의 y 좌표 차이를 계산
            y_difference = left_wrist_y - left_elbow_y
            
            # 왼쪽 손목이 팔꿈치보다 위로 올라갔는지 확인
            if y_difference < 0 and previous_left_hand_y >= left_elbow_y:  # 왼쪽 어깨 -> 왼쪽 팔꿈치로 변경
                save_left_frame_time = time.perf_counter()  # 현재 시간 기록
            # 왼쪽 손목이 팔꿈치보다 아래로 내려갔는지 확인
            elif y_difference > 0 and previous_left_hand_y <= left_elbow_y:  # 왼쪽 어깨 -> 왼쪽 팔꿈치로 변경
                save_left_frame_time = time.perf_counter()  # 현재 시간 기록

        # 왼손 모션이 동작 중인 경우 오른손 모션 인식 방지
        if left_wrist_y != previous_left_hand_y and abs(y_difference) > threshold:  # 왼손 모션이 감지되면
            previous_right_hand_x = None  # 오른손 모션 초기화

        previous_left_hand_y = left_wrist_y  # 이전 왼손 y 좌표 업데이트

    frame_count += 1  # 프레임 카운터 증가

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
        smoothed_mask_resized = cv2.resize(smoothed_mask, (frame.shape[1], frame.shape[0]))
        fg[smoothed_mask_resized == 0] = 0

        # 배경 이미지 로드 및 크기 조정
        background_image = cv2.imread(BACKGROUND_FILES[current_background])
        background_image = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))
        background_image[smoothed_mask_resized == 1] = 0
        new_image = alpha_blending(fg, background_image, smoothed_mask_resized)

        # 화면에 표시
        cv2.imshow('Background Changed', new_image)  # 모델 출력 부분 주석 처리

        # 3초 후에 화면 저장 
        if save_left_frame_time is not None and time.perf_counter() - save_left_frame_time >= 3:
            timestamp = time.strftime("%Y%m%d")  # 현재 날짜
            save_path = f"/home/intel/openvino/mini_project/{timestamp}_{save_sequence}.png"  # 원하는 경로로 변경
            cv2.imwrite(save_path, new_image)  # 이미지 저장
            print(f"저장 완료: {save_path}")
            save_left_frame_time = None  # 저장 후 시간 초기화
            save_sequence += 1  # 순차적으로 저장한 순번 증가
            
        # 남은 타임을 1초마다 출력
        if save_left_frame_time is not None:
            remaining_time = 3 - (time.perf_counter() - save_left_frame_time)
            if remaining_time > 0:
                print(f"남은 시간: {int(remaining_time)}초")  # 소수점 제거

    # 'ESC' 키로 종료
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# 웹캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
