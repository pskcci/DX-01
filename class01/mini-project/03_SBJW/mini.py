# 가칭; 프로젝트 쥬엘(후크선장 앵무새)
# 합체 v3 / 컵 감지 기능 있던거 병합.
import cv2
import numpy as np
import time
import requests
from pathlib import Path
import openvino as ov
import openvino.properties as props
from notebook_utils import download_file, load_image
import notebook_utils as utils
import openvino.properties.hint as hints
import torch
from typing import Tuple
from numpy.lib.stride_tricks import as_strided
import collections
from IPython import display
import threading
import queue
import warnings



midas_model_path = None
pose_model_path = None
model2 = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # yolov5s 모델 사용

#region

# 심도모델
def download_midas_model():
    """Download and set up the MiDaS Depth Estimation model."""
    # A directory where the model will be stored
    model_folder = Path("model/midas")
    model_folder.mkdir(parents=True, exist_ok=True)

    ir_model_url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/depth-estimation-midas/FP32/"
    ir_model_name_xml = "MiDaS_small.xml"
    ir_model_name_bin = "MiDaS_small.bin"

    # Download model files if not already present
    if not (model_folder / ir_model_name_xml).exists():
        download_file(ir_model_url + ir_model_name_xml, filename=ir_model_name_xml, directory=model_folder)
    if not (model_folder / ir_model_name_bin).exists():
        download_file(ir_model_url + ir_model_name_bin, filename=ir_model_name_bin, directory=model_folder)

    model_xml_path = model_folder / ir_model_name_xml
    print(f"MiDaS model downloaded at: {model_xml_path}")
    return model_xml_path

# 포즈모델
def download_pose_estimation_model():
    """Download and set up the Human Pose Estimation model."""
    # A directory where the model will be stored
    base_model_dir = Path("model/pose_estimation")
    base_model_dir.mkdir(parents=True, exist_ok=True)

    model_name = "human-pose-estimation-0001"
    precision = "FP16-INT8"

    model_path = base_model_dir / "intel" / model_name / precision / f"{model_name}.xml"
    
    # Download model files if not already present
    if not model_path.exists():
        model_url_dir = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/{model_name}/{precision}/"
        download_file(model_url_dir + model_name + ".xml", model_path.name, model_path.parent)
        download_file(
            model_url_dir + model_name + ".bin",
            model_path.with_suffix(".bin").name,
            model_path.parent,
        )
    print(f"Pose Estimation model downloaded at: {model_path}")
    return model_path

# 이 함수 실행하면 모델이 있는지 판단하고 없으면 설치함.
# Check if models exist; if not, download them
def setup_models():
    """Download models if they are not already downloaded."""
    global midas_model_path, pose_model_path  # 전역 변수 선언
    print("Checking and downloading MiDaS model...")
    midas_model_path = download_midas_model()  # MiDaS 모델 경로 설정

    print("Checking and downloading Pose Estimation model...")
    pose_model_path = download_pose_estimation_model()  # 포즈 모델 경로 설정

    print("\nModels are ready for use!")



# WebcamProcessor 클래스
class WebcamProcessor:
    def __init__(self, camera_id=0, frame_width=1280, frame_height=720):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError("웹캠을 열 수 없습니다.")
        
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.current_frame = None

    def read_frame(self):
        """웹캠으로부터 프레임을 읽어옵니다."""
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("웹캠에서 영상을 읽을 수 없습니다.")
        self.current_frame = frame
        return frame

    def release(self):
        """웹캠 자원을 해제합니다."""
        self.cap.release()

# DepthProcessor 클래스
class DepthProcessor:
    def __init__(self, compiled_model, input_key, output_key):
        self.compiled_model = compiled_model
        self.input_key = input_key
        self.output_key = output_key

    def process_frame(self, frame):
        """주어진 프레임에서 뎁스 결과를 생성합니다."""
        resized_frame = cv2.resize(frame, (self.input_key.shape[2], self.input_key.shape[3]))
        input_image = np.expand_dims(np.transpose(resized_frame, (2, 0, 1)), 0)
        result = self.compiled_model([input_image])[self.output_key]
        return result

    def visualize_result(self, result):
        """뎁스 결과를 시각화합니다."""
        result_frame = self.convert_result_to_image(result)
        return result_frame

    @staticmethod
    def normalize_minmax(data):
        """뎁스 데이터를 정규화합니다."""
        return (data - data.min()) / (data.max() - data.min())

    def convert_result_to_image(self, result, colormap="viridis"):
        """뎁스 결과를 컬러맵으로 변환합니다."""
        import matplotlib.cm
        cmap = matplotlib.colormaps[colormap]
        result = result.squeeze(0)
        result = self.normalize_minmax(result)
        result = cmap(result)[:, :, :3] * 255
        result = result.astype(np.uint8)
        return result


def process_depth_sections(depth_map, num_rows=4, num_cols=4, threshold=0.8):
    """
    16등분 섹션의 대표 심도 값을 계산하고, 기준값 이상의 섹션을 왼쪽과 오른쪽으로 나눕니다.
    
    Args:
        depth_map: 정규화된 뎁스 맵 (0.0 ~ 1.0)
        num_rows: 세로 섹션 수
        num_cols: 가로 섹션 수
        threshold: 기준 심도 값
    
    Returns:
        decision: "Avoid to Right", "Avoid to Left", or "Balanced"
    """
    h, w = depth_map.shape
    section_height = h // num_rows
    section_width = w // num_cols
    
    left_count = 0
    right_count = 0

    for row in range(num_rows):
        for col in range(num_cols):
            # 섹션 좌표 계산
            y1, y2 = row * section_height, (row + 1) * section_height
            x1, x2 = col * section_width, (col + 1) * section_width

            # 섹션 영역 추출
            section = depth_map[y1:y2, x1:x2]
            
            # 섹션의 평균 심도 계산
            mean_depth = section.mean()
            
            # 기준값 이상인지 확인
            if mean_depth >= threshold:
                if col < num_cols // 2:
                    left_count += 1  # 왼쪽 영역
                else:
                    right_count += 1  # 오른쪽 영역

    # 결정 논리
    if left_count > right_count:
        return "Avoid to Right"
    elif right_count > left_count:
        return "Avoid to Left"
    else:
        return "Balanced"
    
def display_depth_sections(image, depth_map, num_rows=4, num_cols=4, output_width=1280, output_height=720):
    """
    화면을 num_rows x num_cols 섹션으로 나누고, 각 섹션의 대표 심도 값을 계산하여 출력 크기에 맞춰 표시합니다.
    
    Args:
        image: 원본 프레임 (BGR 이미지)
        depth_map: 정규화된 뎁스 맵 (0.0 ~ 1.0)
        num_rows: 세로 섹션 수
        num_cols: 가로 섹션 수
        output_width: 출력 화면의 폭
        output_height: 출력 화면의 높이
    """
    # 출력 해상도로 크기 조정
    image = cv2.resize(image, (output_width, output_height))
    depth_map = cv2.resize(depth_map, (output_width, output_height))

    # 섹션 크기 계산
    section_height = output_height // num_rows
    section_width = output_width // num_cols

    for row in range(num_rows):
        for col in range(num_cols):
            # 섹션 좌표 계산
            y1, y2 = row * section_height, (row + 1) * section_height
            x1, x2 = col * section_width, (col + 1) * section_width

            # 섹션 영역 추출
            section = depth_map[y1:y2, x1:x2]
            
            # 섹션의 평균 심도 계산
            mean_depth = section.mean()

            # 섹션에 심도 값 표시
            cv2.putText(
                image,
                f"{mean_depth:.2f}",
                (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # 흰색 텍스트
                1,
                cv2.LINE_AA
            )
            
            # 섹션 경계선 그리기
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)  # 초록색 경계선

    return image
#endregion


#region
device = utils.device_widget()
device


def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        exit()

    # # 비디오 저장 설정
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_path, fourcc, 30, (640, 480))  # Adjust frame size as needed


############# sector 2 ###########
import openvino.properties.hint as hints

# Initialize OpenVINO Runtime
core = ov.Core()
# Read the network from a file.
setup_models()
model = core.read_model(pose_model_path)
# Let the AUTO device decide where to load the model (you can use CPU, GPU as well).
compiled_model = core.compile_model(model=model, device_name=device.value, config={hints.performance_mode(): hints.PerformanceMode.LATENCY})
# Get the input and output names of nodes.
input_layer = compiled_model.input(0)
output_layers = compiled_model.outputs

# Get the input size.
height, width = list(input_layer.shape)[2:]
# sector 3
# 입력 레이어 출력 레이어 
input_layer.any_name, [o.any_name for o in output_layers]


# # 웹캠화면 객체화
# region
# class WebcamProcessor:
#     def __init__(self, camera_id=0, frame_width=1280, frame_height=720):
#         self.cap = cv2.VideoCapture(camera_id)
#         if not self.cap.isOpened():
#             raise ValueError("웹캠을 열 수 없습니다.")
        
#         self.frame_width = frame_width
#         self.frame_height = frame_height
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
#         self.current_frame = None

#     def read_frame(self):
#         """웹캠으로부터 프레임을 읽어옵니다."""
#         ret, frame = self.cap.read()
#         if not ret:
#             raise ValueError("웹캠에서 영상을 읽을 수 없습니다.")
#         self.current_frame = frame
#         return frame

#     def release(self):
#         """웹캠 자원을 해제합니다."""
#         self.cap.release()
# endregion


# sector 4
# 오픈포즈 디코더를 통해 신경망의 원시 결과를 자세 추정값으로 변환하기(주피터 코드)
# code from https://github.com/openvinotoolkit/open_model_zoo/blob/9296a3712069e688fe64ea02367466122c8e8a3b/demos/common/python/models/open_pose.py#L135
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

    
    def __init__(
        self,
        num_joints=18,
        skeleton=BODY_PARTS_KPT_IDS,
        paf_indices=BODY_PARTS_PAF_IDS,
        max_points=100,
        score_threshold=0.1,
        min_paf_alignment_score=0.05,
        delta=0.5,
    ):
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
            # Filter low-score points.
            mask = masks[0, k]
            x = xs[0, k][mask].ravel()
            y = ys[0, k][mask].ravel()
            score = scores[0, k][mask].ravel()
            n = len(x)
            if n == 0:
                all_keypoints.append(np.empty((0, 4), dtype=np.float32))
                continue
            # Apply quarter offset to improve localization accuracy.
            x, y = self.refine(heatmaps[0, k], x, y)
            np.clip(x, 0, w - 1, out=x)
            np.clip(y, 0, h - 1, out=y)
            # Pack resulting points.
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
        # Get positions with top scores.
        ind = heatmaps.argpartition(-self.max_points, axis=2)[:, :, -self.max_points :]
        scores = np.take_along_axis(heatmaps, ind, axis=2)
        # Keep top scores sorted.
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

    def update_poses(
        self,
        kpt_a_id,
        kpt_b_id,
        all_keypoints,
        connections,
        pose_entries,
        pose_entry_size,
    ):
        for connection in connections:
            pose_a_idx = -1
            pose_b_idx = -1
            for j, pose in enumerate(pose_entries):
                if pose[kpt_a_id] == connection[0]:
                    pose_a_idx = j
                if pose[kpt_b_id] == connection[1]:
                    pose_b_idx = j
            if pose_a_idx < 0 and pose_b_idx < 0:
                # Create new pose entry.
                pose_entry = np.full(pose_entry_size, -1, dtype=np.float32)
                pose_entry[kpt_a_id] = connection[0]
                pose_entry[kpt_b_id] = connection[1]
                pose_entry[-1] = 2
                pose_entry[-2] = np.sum(all_keypoints[connection[0:2], 2]) + connection[2]
                pose_entries.append(pose_entry)
            elif pose_a_idx >= 0 and pose_b_idx >= 0 and pose_a_idx != pose_b_idx:
                # Merge two poses are disjoint merge them, otherwise ignore connection.
                pose_a = pose_entries[pose_a_idx]
                pose_b = pose_entries[pose_b_idx]
                if self.is_disjoint(pose_a, pose_b):
                    pose_a += pose_b
                    pose_a[:-2] += 1
                    pose_a[-2] += connection[2]
                    del pose_entries[pose_b_idx]
            elif pose_a_idx >= 0 and pose_b_idx >= 0:
                # Adjust score of a pose.
                pose_entries[pose_a_idx][-2] += connection[2]
            elif pose_a_idx >= 0:
                # Add a new limb into pose.
                pose = pose_entries[pose_a_idx]
                if pose[kpt_b_id] < 0:
                    pose[-2] += all_keypoints[connection[1], 2]
                pose[kpt_b_id] = connection[1]
                pose[-2] += connection[2]
                pose[-1] += 1
            elif pose_b_idx >= 0:
                # Add a new limb into pose.
                pose = pose_entries[pose_b_idx]
                if pose[kpt_a_id] < 0:
                    pose[-2] += all_keypoints[connection[0], 2]
                pose[kpt_a_id] = connection[0]
                pose[-2] += connection[2]
                pose[-1] += 1
        return pose_entries

    @staticmethod
    def connections_nms(a_idx, b_idx, affinity_scores):
        # From all retrieved connections that share starting/ending keypoints leave only the top-scoring ones.
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
        # For every limb.
        for part_id, paf_channel in enumerate(self.paf_indices):
            kpt_a_id, kpt_b_id = self.skeleton[part_id]
            kpts_a = all_keypoints_by_type[kpt_a_id]
            kpts_b = all_keypoints_by_type[kpt_b_id]
            n = len(kpts_a)
            m = len(kpts_b)
            if n == 0 or m == 0:
                continue

            # Get vectors between all pairs of keypoints, i.e. candidate limb vectors.
            a = kpts_a[:, :2]
            a = np.broadcast_to(a[None], (m, n, 2))
            b = kpts_b[:, :2]
            vec_raw = (b[:, None, :] - a).reshape(-1, 1, 2)

            # Sample points along every candidate limb vector.
            steps = 1 / (self.points_per_limb - 1) * vec_raw
            points = steps * self.grid + a.reshape(-1, 1, 2)
            points = points.round().astype(dtype=np.int32)
            x = points[..., 0].ravel()
            y = points[..., 1].ravel()

            # Compute affinity score between candidate limb vectors and part affinity field.
            part_pafs = pafs[0, :, :, paf_channel : paf_channel + 2]
            field = part_pafs[y, x].reshape(-1, self.points_per_limb, 2)
            vec_norm = np.linalg.norm(vec_raw, ord=2, axis=-1, keepdims=True)
            vec = vec_raw / (vec_norm + 1e-6)
            affinity_scores = (field * vec).sum(-1).reshape(-1, self.points_per_limb)
            valid_affinity_scores = affinity_scores > self.min_paf_alignment_score
            valid_num = valid_affinity_scores.sum(1)
            affinity_scores = (affinity_scores * valid_affinity_scores).sum(1) / (valid_num + 1e-6)
            success_ratio = valid_num / self.points_per_limb

            # Get a list of limbs according to the obtained affinity score.
            valid_limbs = np.where(np.logical_and(affinity_scores > 0, success_ratio > 0.8))[0]
            if len(valid_limbs) == 0:
                continue
            b_idx, a_idx = np.divmod(valid_limbs, n)
            affinity_scores = affinity_scores[valid_limbs]

            # Suppress incompatible connections.
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

            # Update poses with new connections.
            pose_entries = self.update_poses(
                kpt_a_id,
                kpt_b_id,
                all_keypoints,
                connections,
                pose_entries,
                pose_entry_size,
            )

        # Remove poses with not enough points.
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

# 2D pooling in numpy (from: https://stackoverflow.com/a/54966908/1624463)
def pool2d(A, kernel_size, stride, padding, pool_mode="max"):

    # Padding
    A = np.pad(A, padding, mode="constant")

    # Window view of A
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

    # Return the result of pooling.
    if pool_mode == "max":
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == "avg":
        return A_w.mean(axis=(1, 2)).reshape(output_shape)


# non maximum suppression
def heatmap_nms(heatmaps, pooled_heatmaps):
    return heatmaps * (heatmaps == pooled_heatmaps)


# Get poses from results.
def process_results(img, pafs, heatmaps):
    # This processing comes from
    # https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/python/models/open_pose.py
    pooled_heatmaps = np.array([[pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]])
    nms_heatmaps = heatmap_nms(heatmaps, pooled_heatmaps)

    # Decode poses.
    poses, scores = decoder(heatmaps, nms_heatmaps, pafs)
    output_shape = list(compiled_model.output(index=0).partial_shape)
    output_scale = (
        img.shape[1] / output_shape[3].get_length(),
        img.shape[0] / output_shape[2].get_length(),
    )
    # Multiply coordinates by a scaling factor.
    poses[:, :, :2] *= output_scale
    return poses, scores

colors = (
    # 이 친구들의 BGR 값을 0 0 0 으로 바꾸다 보면 어디 점이 어딘지 알수 있다.
    (255, 0, 0), #오른눈
    (255, 0, 255), # 코
    (170, 0, 0), # 왼눈
    (255, 0, 85), # 오른귀
    (255, 0, 170), # 왼귀
    (85, 255, 0), # 오른어깨
    (255, 170, 0), # 왼어깨
    (0, 255, 0), # 오른팔꿈치
    (255, 255, 0), # 왼팔꿈치
    (0, 255, 85), # 오른손목
    (170, 255, 0), # 왼손목
    (0, 85, 255), # 오른골반
    (0, 255, 170), #왼골반
    (0, 0, 255), #오른무릎
    (0, 255, 255), #왼무릎
    (85, 0, 255), #오른발목
    (0, 170, 255), #왼발목
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

## todo : 글자 floating 위치 수정하고 투명도 조절되면 조절하기
def draw_poses_with_description(img, poses, point_score_threshold, skeleton=default_skeleton):
    if poses.size == 0:
        return img

    img_limbs = np.copy(img)
    for pose in poses:
        points = pose[:, :2].astype(np.int32)
        points_scores = pose[:, 2]

        # 양팔 든 상태를 인식
        if points_scores[9] > point_score_threshold and points_scores[10] > point_score_threshold:
            right_wrist = points[9]  # 오른손목
            left_wrist = points[10]  # 왼손목
            right_shoulder = points[5]  # 오른어깨
            left_shoulder = points[6]  # 왼어깨

            # 양팔을 든 상태 조건: 손목의 Y 좌표가 어깨의 Y 좌표보다 작음(위쪽)
            if (
                right_wrist[1] < right_shoulder[1]  # 오른손목이 오른어깨보다 위
                and left_wrist[1] < left_shoulder[1]  # 왼손목이 왼어깨보다 위
            ):
                # "Vingo" 텍스트 출력
                cv2.putText(
                    img,
                    "Vingo",
                    (img.shape[1] // 2, img.shape[0] // 4),  # 화면 중앙 상단에 출력
                    cv2.FONT_HERSHEY_SIMPLEX,
                    5,  # 글자 크기
                    (0, 0, 0),  # 초록색
                    13,  # 글자 두께
                    cv2.LINE_AA,
                )

            # 왼쪽 손목이 어깨보다 위에 있을 때 "Left" 출력
            if left_wrist[1] < left_shoulder[1]:  # Y 좌표가 작으면 더 위에 있음
                cv2.putText(
                    img,
                    "Left",
                    (50, 50),  # 화면 왼쪽 상단
                    cv2.FONT_HERSHEY_SIMPLEX,
                    5,  # 글자 크기
                    (255, 0, 0),  # 파란색
                    13,  # 글자 두께
                    cv2.LINE_AA,
                )

            # 오른쪽 손목이 어깨보다 위에 있을 때 "Right" 출력
            if right_wrist[1] < right_shoulder[1]:
                cv2.putText(
                    img,
                    "Right",
                    (img.shape[1] - 200, 50),  # 화면 오른쪽 상단
                    cv2.FONT_HERSHEY_SIMPLEX,
                    5,  # 글자 크기
                    (0, 0, 255),  # 빨간색
                    13,  # 글자 두께
                    cv2.LINE_AA,
                )

        # 관절 포인트 및 스켈레톤 그리기
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img, tuple(p), 6, colors[i], 6)

        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                cv2.line(
                    img_limbs,
                    tuple(points[i]),
                    tuple(points[j]),
                    color=colors[j],
                    thickness=4,
                )

    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    return img


# Main processing function to run pose estimation.
def run_pose_estimation(frame, flip=True, use_popup=False, skip_first_frames=0):
    # OpenVINO 모델 처리와 관련된 변수 설정
    pafs_output_key = compiled_model.output("Mconv7_stage2_L1")
    heatmaps_output_key = compiled_model.output("Mconv7_stage2_L2")
    processing_times = collections.deque()

    try:
        # Resize the image and change dims to fit neural network input.
        # Resize frame to fit neural network input (320x240 or 640x480)
        scale = 1280 / max(frame.shape)
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        input_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        input_img = input_img.transpose((2, 0, 1))[np.newaxis, ...]

        # Measure processing time
        start_time = time.time()

        # Get results from OpenVINO compiled model
        results = compiled_model([input_img])
        stop_time = time.time()

        # Extract pose information from results
        pafs = results[pafs_output_key]
        heatmaps = results[heatmaps_output_key]
        poses, scores = process_results(frame, pafs, heatmaps)

        # Draw poses on the frame
        frame = draw_poses_with_description(frame, poses, 0.1)

        # Calculate FPS
        processing_times.append(stop_time - start_time)
        if len(processing_times) > 200:
            processing_times.popleft()

        _, f_width = frame.shape[:2]
        processing_time = np.mean(processing_times) * 1000
        fps = 1000 / processing_time
        cv2.putText(
            frame,
            f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
            (20, 40),
            cv2.FONT_HERSHEY_COMPLEX,
            f_width / 1000,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        # Display frame with pose
        if use_popup:
            cv2.imshow("Pose Estimation", frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key to exit
                return
        else:
            # If not using a popup, display frame in notebook (for Jupyter)
            _, encoded_img = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 90])
            i = display.Image(data=encoded_img)
            display.clear_output(wait=True)
            # <IPython.core.display.Image object> 이걸 터미널에 계속 출력시키는 원인
            # 일단 억제해도 코드 원하는대로 돌아가는 거 확인함
            # display.display(i)

    except Exception as e:
        print(f"Error in pose estimation: {e}")

def model_init(model_path: str) -> Tuple:
    model2 = core.read_model(model2=model_path)
    compiled_model = core.compile_model(model2=model, device_name=device.value)
    input_keys = compiled_model.input(0)
    output_keys = compiled_model.output(0)
    return input_keys, output_keys, compiled_model

beverage_classes = ["cup", "can", "bottle"]

warnings.filterwarnings("ignore", category=FutureWarning)
# YOLO 모델을 사용하여 음료 객체를 감지
def detect_beverages(frame):
    results = model2(frame)  # 이미지에서 객체 검출
    detections = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, confidence, class


    # 감지된 객체가 음료 관련 클래스인 경우만 필터링
    beverage_detections = []
    for *box, conf, cls in detections:
        if model2.names[int(cls)] in beverage_classes:  # 음료 관련 클래스일 경우
            beverage_detections.append([*box, conf, cls])
    
    return beverage_detections

# 음료 객체의 결과를 이미지에 그리는 함수
def draw_beverages(frame, detections):
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        label = f"{model2.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 바운딩 박스 그리기
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 레이블 텍스트 추가
    return frame

# region
'''''
동영상으로 컵 객체 확인 할 때 쓰는 코드...?
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        exit()

    # 비디오 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (640, 480))  # Adjust frame size as needed

    # Read frames from the video
    while cap.isOpened():
        ret, frame = cap.read()  # 비디오 프레임 읽기
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # 음료 객체 감지
        detections = detect_beverages(frame)
        frame = draw_beverages(frame, detections)

        # 객체 검출 결과 화면에 표시
        cv2.imshow("Beverage Detection", frame)
        
        # 결과 비디오에 저장
        out.write(frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
'''
#endregion

# 버리는 기능
#video_path = '/home/intel/openvino/beverage.mp4'  # 비디오 파일 경로
#output_path = 'output/processed_video.mp4'  # 출력 비디오 경로

# process_video(video_path, output_path)
# 이 def로 영상처리 조진다 
def process_webcam(webcam_processor):
    """
    WebcamProcessor 객체를 사용하여 실시간 웹캠 영상을 처리합니다.
    """
    try:
        while True:
            # WebcamProcessor 객체로부터 프레임 읽기
            frame = webcam_processor.read_frame()

            # 포즈 추정 처리
            run_pose_estimation(frame, flip=False, use_popup=True)

            # 음료 객체 감지 및 결과 표시
            detections = detect_beverages(frame)  # 음료 객체 감지
            frame = draw_beverages(frame, detections)  # 감지된 객체 그리기

            # 화면에 결과 표시
            # cv2.imshow("Beverage Detection", frame)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        # 자원 해제
        webcam_processor.release()
        cv2.destroyAllWindows()
#endregion1



# 메인 루프
# Depth estimation 함수
def depth_main(webcam_processor, result_queue):
    global midas_model_path  # 전역 변수 선언
    
    # OpenVINO 모델 초기화
    core = ov.Core()
    cache_folder = Path("cache")
    cache_folder.mkdir(exist_ok=True)
    core.set_property({props.cache_dir(): cache_folder})
    
    # MiDaS 모델 읽기 및 컴파일
    print(f"Using MiDaS model at path: {midas_model_path}")
    model = core.read_model(midas_model_path)  # 전역 변수로 설정된 경로 사용
    compiled_model = core.compile_model(model=model, device_name="GPU.1")
    input_key = compiled_model.input(0)
    output_key = compiled_model.output(0)
    
    # 뎁스 프로세서 초기화
    depth_processor = DepthProcessor(compiled_model, input_key, output_key)

    try:
        while True:
            frame = webcam_processor.read_frame()  # 웹캠에서 프레임 읽기
            depth_result = depth_processor.process_frame(frame)  # 뎁스 처리
            
            # 뎁스 맵 정규화
            depth_map = (depth_result.squeeze(0) - depth_result.min()) / (depth_result.max() - depth_result.min())
            
            # 16등분 섹션 처리 및 결정
            decision = process_depth_sections(depth_map, num_rows=4, num_cols=4, threshold=0.8)
            
            # 시각화된 뎁스 맵
            depth_frame = depth_processor.visualize_result(depth_result)
            
            # 16등분 영역에 심도 정보 표시 (1280x720 크기로 출력)
            display_frame = display_depth_sections(depth_frame.copy(), depth_map, num_rows=4, num_cols=4, output_width=1280, output_height=720)
            
            # 결정 결과 출력
            cv2.putText(
                display_frame,
                decision,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),  # 빨간색 텍스트
                2,
                cv2.LINE_AA
            )

            # 결과를 큐에 넣어 메인 스레드에서 출력
            result_queue.put(('depth', display_frame))
    except KeyboardInterrupt:
        print("Depth estimation stopped.")
    finally:
        webcam_processor.release()

# Pose estimation 및 음료 객체 감지 함수
def run_pose_and_beverage_estimation(webcam_processor, result_queue):
    try:
        while True:
            frame = webcam_processor.read_frame()  # 웹캠에서 프레임 읽기

            # 음료 객체 감지
            detections = detect_beverages(frame)
            frame_with_beverages = draw_beverages(frame, detections)

            # 포즈 추정
            run_pose_estimation(frame_with_beverages, flip=False, use_popup=False)

            # 결과를 큐에 넣어 메인 스레드에서 출력
            result_queue.put(('pose_and_beverage', frame_with_beverages))
    except Exception as e:
        print(f"Error during pose and beverage estimation: {e}")
    finally:
        webcam_processor.release()

def unified_depth_pose_estimation(webcam_processor):
    global midas_model_path, pose_model_path  # 전역 모델 경로 접근

    # OpenVINO Core 초기화
    core = ov.Core()
    cache_folder = Path("cache")
    cache_folder.mkdir(parents=True, exist_ok=True)
    core.set_property({props.cache_dir(): cache_folder})

    # MiDaS 깊이 추정 모델 로드
    model_depth = core.read_model(midas_model_path)
    compiled_model_depth = core.compile_model(model=model_depth, device_name="GPU.1")
    input_key_depth = compiled_model_depth.input(0)
    output_key_depth = compiled_model_depth.output(0)
    depth_processor = DepthProcessor(compiled_model_depth, input_key_depth, output_key_depth)

    # 포즈 추정 모델 로드
    model_pose = core.read_model(pose_model_path)
    compiled_model_pose = core.compile_model(model=model_pose, device_name="GPU.1")
    input_key_pose = compiled_model_pose.input(0)
    pafs_output_key = compiled_model_pose.output("Mconv7_stage2_L1")
    heatmaps_output_key = compiled_model_pose.output("Mconv7_stage2_L2")
    height, width = list(input_key_pose.shape)[2:]

    try:
        while True:
            # 웹캠에서 프레임 읽기
            frame = webcam_processor.read_frame()

            # 성능 향상을 위해 프레임 건너뛰기 (선택 사항)
            if int(time.time() * 10) % 2 != 0:
                continue

            ### 깊이 추정 ###
            depth_result = depth_processor.process_frame(frame)
            depth_map = (depth_result.squeeze(0) - depth_result.min()) / (depth_result.max() - depth_result.min())
            depth_frame = depth_processor.visualize_result(depth_result)

            # 깊이 맵을 16등분하여 분석
            decision = process_depth_sections(depth_map, num_rows=4, num_cols=4, threshold=0.8)

            # 섹션 정보와 함께 깊이 맵 표시 (1280x720 해상도)
            depth_frame_with_sections = display_depth_sections(
                depth_frame.copy(), depth_map, num_rows=4, num_cols=4, output_width=1280, output_height=720
            )

            # 깊이 출력에 결정 텍스트 추가
            cv2.putText(
                depth_frame_with_sections,
                decision,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),  # 빨간색 텍스트
                2,
                cv2.LINE_AA
            )

            ### 포즈 추정 ###
            # 포즈 추정을 위한 입력 준비
            input_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            input_img = input_img.transpose((2, 0, 1))[np.newaxis, ...]

            # 포즈 추정 추론 실행
            results_pose = compiled_model_pose([input_img])
            pafs = results_pose[pafs_output_key]
            heatmaps = results_pose[heatmaps_output_key]
            poses, _ = process_results(frame, pafs, heatmaps)

            # 프레임에 포즈 그리기
            frame_with_poses = draw_poses_with_description(frame, poses, 0.1)

            ### 음료 객체 감지 ###
            detections = detect_beverages(frame_with_poses)
            frame_with_all = draw_beverages(frame_with_poses, detections)

            # 각 손과 컵 객체의 충돌 검사 (영역을 30픽셀 확장하여 조건 완화)
            for pose in poses:
                right_hand = pose[9, :2]  # 오른손목 좌표
                left_hand = pose[10, :2]  # 왼손목 좌표
                if pose[9, 2] > 0.1 or pose[10, 2] > 0.1:  # 각 손이 감지된 경우
                    for *box, conf, cls in detections:
                        x1, y1, x2, y2 = map(int, box)
                        # 손 위치가 컵 영역 내에 있는지 확인 (영역을 30픽셀 확장)
                        if (pose[9, 2] > 0.1 and x1 - 30 <= right_hand[0] <= x2 + 30 and y1 - 30 <= right_hand[1] <= y2 + 30) or \
                            (pose[10, 2] > 0.1 and x1 - 30 <= left_hand[0] <= x2 + 30 and y1 - 30 <= left_hand[1] <= y2 + 30):
                            cv2.putText(
                                frame_with_all,
                                "CATCH!!",
                                (frame.shape[1] // 2, frame.shape[0] // 2),  # 화면 중앙에 출력
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2,  # 글자 크기
                                (0, 255, 255),  # 노란색
                                5,  # 글자 두께
                                cv2.LINE_AA,
                            )
                            print("CATCH!!")  # 터미널에도 'CATCH!!' 출력
                            break

            ### 깊이와 포즈 출력 결합 ###
            # 섹션이 있는 깊이 프레임과 포즈 및 음료 객체가 있는 프레임을 나란히 결합
            combined_frame = np.hstack((cv2.resize(depth_frame_with_sections, (640, 480)), cv2.resize(frame_with_all, (640, 480))))

            # 결합된 프레임 표시
            cv2.imshow("Depth and Pose Estimation", combined_frame)

            # 'q' 키를 눌러 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # CPU 리소스를 해제하기 위한 짧은 지연
            time.sleep(0.01)

    except Exception as e:
        print(f"Error during unified estimation: {e}")
    finally:
        # 웹캠 자원 해제
        webcam_processor.release()
        cv2.destroyAllWindows()

# 통합 추정을 시작하는 메인 함수
def main():
    # 모델 설정 (이미 다운로드되어 있지 않은 경우 다운로드)
    setup_models()

    # 웹캠 프로세서 초기화
    webcam_processor = WebcamProcessor(camera_id=0)

    # 통합 깊이 및 포즈 추정 실행
    unified_depth_pose_estimation(webcam_processor)

if __name__ == "__main__":
    main()
