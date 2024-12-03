import collections
import time
from pathlib import Path
from IPython import display
from numpy.lib.stride_tricks import as_strided
import openvino as ov
import requests
from pathlib import Path
import gc
import openvino as ov
from ultralytics import YOLO
from IPython import display
import cv2
import numpy as np
import openvino.properties.hint as hints
import threading
from queue import Queue



# region 전역변수
# 웹캠 제어 파트
conx, cony = 0, 0
base_model_dir = Path("model")
model_name = "human-pose-estimation-0001"
precision = "FP16-INT8"
model_path = base_model_dir / "intel" / model_name / precision / f"{model_name}.xml"
device = None
core = None
model = None
compiled_model = None
input_layer = None
output_layers = None
height, width = None, None
# 오브젝트 디텍션 파트
det_model_name = "yolov8n"
det_model_path = Path(f"{det_model_name}_openvino_model/{det_model_name}.xml")
det_model = None
det_core = None
det_device = None
# endregion

# region 포즈 디코더 정의
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
# endregion

# region 2D 풀링 관련 및 스켈레톤, 색상 정보
# 2D pooling in numpy (from: https://stackoverflow.com/a/54966908/1624463)
def pool2d(A, kernel_size, stride, padding, pool_mode="max"):
    """
    2D Pooling
    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    """
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
    (255, 0, 0),      # Blue
    (255, 0, 255),    # Magenta 안면 1
    (170, 0, 255),    # Purple 안면 2
    (255, 0, 85),     # Pink
    (255, 0, 170),    # Light Magenta
    (85, 255, 0),     # Green
    (255, 170, 0),    # Orange
    (0, 255, 0),      # Green
    (255, 255, 0),    # Yellow
    (0, 0, 0), #(0, 255, 85),     # Light Green 왼손 9
    (0, 0, 0), #(170, 255, 0),    # Lime 오른손 10
    (0, 85, 255),     # Cyan
    (0, 255, 170),    # Light Cyan 
    (0, 0, 255),      # Red
    (0, 255, 255),    # Yellow Green
    (85, 0, 255),     # Violet
    (0, 170, 255),    # Sky Blue
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
# endregion

# region 포즈 그리기
def draw_poses(img, poses, point_score_threshold, skeleton=default_skeleton):
    if poses.size == 0:
        return img

    img_limbs = np.copy(img)
    for pose in poses:
        points = pose[:, :2].astype(np.int32)
        points_scores = pose[:, 2]
        # Draw joints.
        # 양 손목으로 지정된 포인트와 머리 포인트 저장용 좌표 초기화
        p9, p10, p2, p3 = None, None, None, None
        # Draw joints and check condition
        for i, (p, v) in enumerate(zip(points, points_scores)):
            # Draw the joint on the image
            cv2.circle(img, tuple(p), 5, colors[i], 5)
            if v > point_score_threshold:
                if i == 9:
                    p9 = p
                elif i == 10:
                    p10 = p
        # 중심 좌표와 사각형 크기 설정
        sq_x, sq_y = 300, 500
        sq_size = 100
        # 사각형 좌표 계산
        offset = sq_size // 2
        rectangles = {
            "2": ((sq_x - offset, sq_y + offset), (sq_x + offset, sq_y + 3 * offset)),  # 빨간색
            "4": ((sq_x - 3 * offset, sq_y - offset), (sq_x - offset, sq_y + offset)),  # 빨간색
            "6": ((sq_x + offset, sq_y - offset), (sq_x + 3 * offset, sq_y + offset)),  # 빨간색
            "8": ((sq_x - offset, sq_y - 3 * offset), (sq_x + offset, sq_y - offset)),  # 빨간색
            "1": ((sq_x - 3 * offset, sq_y + offset), (sq_x - offset, sq_y + 3 * offset)),  # 주황색
            "3": ((sq_x + offset, sq_y + offset), (sq_x + 3 * offset, sq_y + 3 * offset)),  # 주황색
            "7": ((sq_x - 3 * offset, sq_y - 3 * offset), (sq_x - offset, sq_y - offset)),  # 주황색
            "9": ((sq_x + offset, sq_y - 3 * offset), (sq_x + 3 * offset, sq_y - offset)),  # 주황색
            "bt": ((1000 - offset, 500 - offset), (1000 + offset, 500 + offset))  # 초록색 사각형
        }
        # 사각형 그리기
        for key, (start, end) in rectangles.items():
            color = (0, 0, 255) if key in ["2", "4", "6", "8"] else (0, 165, 255)  # 빨간색 또는 주황색
            if key == "bt":
                color = (0, 255, 0)
            cv2.rectangle(img, start, end, color, -1)

        global conx, cony
        # 좌표가 각 사각형 내부에 있는지 확인
        if p10 is not None:
            if rectangles["2"][0][0] <= p10[0] <= rectangles["2"][1][0] and rectangles["2"][0][1] <= p10[1] <= rectangles["2"][1][1]:
                cony += 10  # 2 영역: cony 증가
            elif rectangles["8"][0][0] <= p10[0] <= rectangles["8"][1][0] and rectangles["8"][0][1] <= p10[1] <= rectangles["8"][1][1]:
                cony -= 10  # 8 영역: cony 감소
            elif rectangles["4"][0][0] <= p10[0] <= rectangles["4"][1][0] and rectangles["4"][0][1] <= p10[1] <= rectangles["4"][1][1]:
                conx -= 10  # 4 영역: conx 감소
            elif rectangles["6"][0][0] <= p10[0] <= rectangles["6"][1][0] and rectangles["6"][0][1] <= p10[1] <= rectangles["6"][1][1]:
                conx += 10  # 6 영역: conx 증가
            elif rectangles["1"][0][0] <= p10[0] <= rectangles["1"][1][0] and rectangles["1"][0][1] <= p10[1] <= rectangles["1"][1][1]:
                conx -= 7  # 1 영역: 대각선 위 왼쪽 이동
                cony += 7
            elif rectangles["3"][0][0] <= p10[0] <= rectangles["3"][1][0] and rectangles["3"][0][1] <= p10[1] <= rectangles["3"][1][1]:
                conx += 7  # 3 영역: 대각선 위 오른쪽 이동
                cony += 7
            elif rectangles["7"][0][0] <= p10[0] <= rectangles["7"][1][0] and rectangles["7"][0][1] <= p10[1] <= rectangles["7"][1][1]:
                conx -= 7  # 7 영역: 대각선 아래 왼쪽 이동
                cony -= 7
            elif rectangles["9"][0][0] <= p10[0] <= rectangles["9"][1][0] and rectangles["9"][0][1] <= p10[1] <= rectangles["9"][1][1]:
                conx += 7  # 9 영역: 대각선 아래 오른쪽 이동
                cony -= 7

        if p9 is not None:
            if rectangles["bt"][0][0] <= p9[0] <= rectangles["bt"][1][0] and rectangles["bt"][0][1] <= p9[1] <= rectangles["bt"][1][1]:
                conx = 0
                cony = 0
            else:
                pass

        # Draw limbs.
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                cv2.line(
                    img_limbs,
                    tuple(points[i]),
                    tuple(points[j]),
                    color=colors[j],
                    thickness=4,
                )
    cv2.addWeighted(img, 0.5, img_limbs, 0.5, 0, dst=img)
    return img
# endregion

# region 포즈 에스티메이션
def myPoseEetimation(image, source=0, flip=False, use_popup=False, skip_first_frames=0):
    pafs_output_key = compiled_model.output("Mconv7_stage2_L1")
    heatmaps_output_key = compiled_model.output("Mconv7_stage2_L2")
    scale = 1280 / max(image.shape)
    re_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    input_img = cv2.resize(re_image, (width, height), interpolation=cv2.INTER_AREA)
    input_img = input_img.transpose((2, 0, 1))[np.newaxis, ...]
    # Measure processing time.
    start_time = time.time()
    # Get results.
    results = compiled_model([input_img])
    stop_time = time.time()
    pafs = results[pafs_output_key]
    heatmaps = results[heatmaps_output_key]
    # Get poses from network results.
    poses, scores = process_results(re_image, pafs, heatmaps)
    # Draw poses on a frame.
    pe_image = draw_poses(re_image, poses, 0.1)
    processing_times = collections.deque()

    processing_times.append(stop_time - start_time)
    # Use processing times from last 200 frames.
    if len(processing_times) > 200:
        processing_times.popleft()

    _, f_width = pe_image.shape[:2]
    # mean processing time [ms]
    processing_time = np.mean(processing_times) * 1000
    fps = 1000 / processing_time
    return pe_image
def reverseImage(image):
    """
    좌우 반전된 이미지를 반환하는 함수.
    
    Args:
        image (numpy.ndarray): 원본 이미지.
    
    Returns:
        numpy.ndarray: 좌우 반전된 이미지.
    """
    # 이미지 좌우 반전
    reversed_image = cv2.flip(image, 1)
    return reversed_image
# endregion

# region 포즈 에스티메이션 환경설정
def setEnviroments_pe():
    global device, core, model, compiled_model, input_layer, output_layers, height, width

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)
    import notebook_utils as utils

    # 모델이 존재하지 않으면 다운로드
    if not model_path.exists():
        model_url_dir = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/{model_name}/{precision}/"
        utils.download_file(model_url_dir + model_name + ".xml", model_path.name, model_path.parent)
        utils.download_file(
            model_url_dir + model_name + ".bin",
            model_path.with_suffix(".bin").name,
            model_path.parent,
        )

    # 장치 선택 (예: CPU, GPU 등)
    device = utils.device_widget()
    device = "GPU.1"
    # OpenVINO Runtime 초기화
    core = ov.Core()

    # 모델 파일에서 네트워크 읽기
    model = core.read_model(model_path)

    # 장치 이름에 따라 모델 컴파일
    compiled_model = core.compile_model(model=model, device_name=device, config={hints.performance_mode(): hints.PerformanceMode.LATENCY})

    # 입력 및 출력 레이어의 이름 얻기
    input_layer = compiled_model.input(0)
    output_layers = compiled_model.outputs

    # 입력 크기 얻기
    height, width = list(input_layer.shape)[2:]

    # 레이어 이름 출력
    print(input_layer.any_name, [o.any_name for o in output_layers])
# endregion

# region 오브젝트 디텍션 환경설정
def setEnviroments_de():
    global det_core, det_model, det_device, det_model_path
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)
    import notebook_utils as utils
    # 모델이 존재하지 않으면 다운로드
    if not det_model_path.exists():
        pt_model = YOLO(f"{det_model_name}.pt")
        pt_model.export(format="openvino", dynamic=True, half=True)
        del pt_model
        gc.collect()

    # OpenVINO 초기화
    det_core = ov.Core()

    # 장치 선택
    det_device = utils.device_widget()
    det_device = "GPU.0"

    # 모델 로드
    det_model = load_model(det_model_path, det_device)

# 모델 로드 함수: load_model
def load_model(det_model_path, device):
    compiled_model = compile_model(det_model_path, device)
    det_model = YOLO(det_model_path.parent, task="detect")

    if det_model.predictor is None:
        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}  # method defaults
        args = {**det_model.overrides, **custom}
        det_model.predictor = det_model._smart_load("predictor")(overrides=args, _callbacks=det_model.callbacks)
        det_model.predictor.setup_model(model=det_model.model)

    det_model.predictor.model.ov_compiled_model = compiled_model
    return det_model

# 모델 컴파일 함수: compile_model
def compile_model(det_model_path, device):
    det_ov_model = core.read_model(det_model_path)

    ov_config = {}
    if device != "CPU":
        det_ov_model.reshape({0: [1, 3, 640, 640]})
    if "GPU" in device or ("AUTO" in device and "GPU" in core.available_devices):
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    det_compiled_model = core.compile_model(det_ov_model, device, ov_config)
    return det_compiled_model
# endregion

# region 오브젝트 디텍션
def objectDetection(img):
    global det_model
    # 이미지로부터 예측 실행
    input_image = np.array(img)
    detections = det_model(input_image, verbose=False)
    det_img = detections[0].plot()
    return det_img
# endregion


# region 메인
def controlCamera(source):
    import threading
    from queue import Queue
    
    # 웹캠 또는 비디오 파일 열기
    cap = cv2.VideoCapture(source)
    global conx, cony
    

    
    # 큐 생성
    frame_queue = Queue(maxsize=2)
    pose_queue = Queue(maxsize=2)
    detection_queue = Queue(maxsize=2)

    def pose_estimation_thread():
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                pose_frame = myPoseEetimation(frame, source=source, flip=isinstance(source, int), use_popup=False, **additional_options)
                pose_frame = reverseImage(pose_frame)
                pose_queue.put(pose_frame)

    def object_detection_thread():
        while True:
            #이미지는 따로 구해서 절대 경로로 새로 바꿔줄것 
            sightimg = cv2.imread("/home/inteldx/test.jpg")
            if sightimg is not None:
                height, width = sightimg.shape[:2]
                sightimg = cv2.resize(sightimg, (width * 2, height * 2))
                binimg = np.zeros_like(sightimg)
                cv2.rectangle(binimg, (-conx, cony), (-conx+600, cony+400), (255, 255, 255), -1)
                resultimg = cv2.bitwise_and(sightimg, binimg)
                det_frame = objectDetection(resultimg)
                detection_queue.put(det_frame)

    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    # 스레드 시작 
    pose_thread = threading.Thread(target=pose_estimation_thread, daemon=True)
    detect_thread = threading.Thread(target=object_detection_thread, daemon=True) 
    pose_thread.start()
    detect_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:  # 프레임을 읽지 못했을 경우 루프 종료
            print("Video ended or unable to fetch the frame.")
            break

        # 프레임 큐에 추가
        if not frame_queue.full():
            frame_queue.put(frame)

        # 처리된 결과 표시
        if not pose_queue.empty():
            pose_frame = pose_queue.get()
            pose_frame = cv2.pyrDown(pose_frame)
            cv2.imshow('Video Output', pose_frame)

        if not detection_queue.empty():
            det_frame = detection_queue.get()
            cv2.imshow("Test Image", det_frame)

        # ESC 키 입력 시 종료
        if cv2.waitKey(1) & 0xFF == 27:  # 27은 ESC 키의 ASCII 코드
            print("ESC key pressed. Exiting video display.")
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

setEnviroments_pe()
print("finish pe env setting")
setEnviroments_de()
print("finish de env setting")
USE_WEBCAM = True
cam_id = 0
video_file = "https://storage.openvinotoolkit.org/data/test_data/videos/store-aisle-detection.mp4"
source = cam_id if USE_WEBCAM else video_file
additional_options = {"skip_first_frames": 500} if not USE_WEBCAM else {}
print("start")
controlCamera(source)
print("finish")
# endregion