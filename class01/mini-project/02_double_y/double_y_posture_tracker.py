import collections
import time
from pathlib import Path

import cv2
import numpy as np
from IPython import display
from numpy.lib.stride_tricks import as_strided
import openvino as ov

# Fetch notebook_utils module
import requests

r = requests.get(
    url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py", "w").write(r.text)
import notebook_utils as utils

# A directory where the model will be downloaded.
base_model_dir = Path("model")
waist = 0
wrist = 0
twist = 0
onesec = 0
neck = 0
maintime = 0
sec_1 = 0

# The name of the model from Open Model Zoo.
model_name = "human-pose-estimation-0001"
# Selected precision (FP32, FP16, FP16-INT8).
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

device

import openvino.properties.hint as hints


# Initialize OpenVINO Runtime
core = ov.Core()
# Read the network from a file.
model = core.read_model(model_path)
# Let the AUTO device decide where to load the model (you can use CPU, GPU as well).
compiled_model = core.compile_model(model=model, device_name=device.value, config={hints.performance_mode(): hints.PerformanceMode.LATENCY})

# Get the input and output names of nodes.
input_layer = compiled_model.input(0)
output_layers = compiled_model.outputs

# Get the input size.
height, width = list(input_layer.shape)[2:]

input_layer.any_name, [o.any_name for o in output_layers]

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
    (255, 0, 0),        #코
    (255, 0, 255),      #눈1
    (170, 0, 255),      #눈2
    (255, 0, 85),       #귀1
    (255, 0, 170),      #귀2
    (0,0, 255),         #오른쪽어깨
    (0, 255, 0 ),       #왼쪽어깨
    (170, 255, 0),      #오른쪽 팔꿈치
    (255, 255, 0),      #왼쪽팔꿈치     
    (0, 255, 85),       #오른쪽손목
    (170, 255, 0),      #왼쪽손목
    (0, 85, 255),       #오른쪽골반   
    (0, 255, 170),      #왼쪽골반
    (0, 0, 0),          #오른쪽무릎
    (0, 255, 255),      #왼쪽무릎
    (85, 0, 255),       #오른쪽발목
    (0, 170, 255),      #왼쪽발목
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
    global sec_1, neck, onesec, maintime, waist, wrist, twist

    # 사람이 감지되지 않았을 경우 원본 이미지를 반환합니다.
    if poses.size == 0:
        return img

    # 감지된 포즈 중 가장 큰 바운딩 박스를 가진 사람을 선택
    largest_area = 0
    largest_pose = None

    for pose in poses:
        points = pose[:, :2].astype(np.int32)  # 관절 좌표
        points_scores = pose[:, 2]  # 관절 점수
        
        # 바운딩 박스 계산
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        valid_points = points_scores > point_score_threshold  # 신뢰도 기준 필터
        
        if valid_points.sum() > 0:
            min_x, max_x = x_coords[valid_points].min(), x_coords[valid_points].max()
            min_y, max_y = y_coords[valid_points].min(), y_coords[valid_points].max()

            area = (max_x - min_x) * (max_y - min_y)  # 바운딩 박스 영역 계산

            if area > largest_area:
                largest_area = area
                largest_pose = pose

    # 유효한 포즈가 없을 경우 원본 이미지를 반환
    if largest_pose is None:
        return img

    # 가장 큰 포즈를 기반으로 분석 진행
    pose = largest_pose
    points = pose[:, :2].astype(np.int32)  # 관절 좌표
    points_scores = pose[:, 2]  # 관절 점수

    sec = time.time() - maintime
    onesec += sec

    img_limbs = np.copy(img)

    # 주요 관절 변수 초기화
    rsh, lsh, rear, lear = None, None, None, None
    rpvis, lpvis, rank, lank = None, None, None, None
    rwrist, relbow, lwrist, lelbow = None, None, None, None

    for i, (p, v) in enumerate(zip(points, points_scores)):
        if v > point_score_threshold:
            cv2.circle(img, tuple(p), 5, colors[i % len(colors)], 5)  # 관절 점 그리기
            
            # 관절 번호에 따라 변수 할당
            if i == 3: rear = tuple(p)  # 오른쪽 귀
            if i == 4: lear = tuple(p)  # 왼쪽 귀
            if i == 5: rsh = tuple(p)  # 오른쪽 어깨
            if i == 6: lsh = tuple(p)  # 왼쪽 어깨
            if i == 7: relbow = tuple(p)  # 오른쪽 팔꿈치
            if i == 9: rwrist = tuple(p)  # 오른쪽 손목
            if i == 11: rpvis = tuple(p)  # 오른쪽 골반
            if i == 12: lpvis = tuple(p)  # 왼쪽 골반
            if i == 15: rank = tuple(p)  # 오른쪽 발목
            if i == 16: lank = tuple(p)  # 왼쪽 발목

    if onesec >= 1:
        # 자세 문제 분석
        if rsh and rear and abs(rsh[0] - rear[0]) > 40:  # 오른쪽 거북목
            neck += 1

        if lsh and lear and abs(lsh[0] - lear[0]) > 40:  # 왼쪽 거북목
            neck += 1

        if rsh and rpvis and abs(rsh[0] - rpvis[0]) > 80:  # 오른쪽 허리
            waist += 1

        if lsh and lpvis and abs(lsh[0] - lpvis[0]) > 80:  # 왼쪽 허리
            waist += 1

        if rank and lank and abs(rank[1] - lank[1]) > 30:  # 다리 꼬기
            twist += 1

        if rwrist and relbow and (relbow[1] - rwrist[1]) > 50:  # 오른쪽 손목
            wrist += 1

        if lwrist and lelbow and (lelbow[1] - lwrist[1]) > 50:  # 왼쪽 손목
            wrist += 1

        sec_1 += 1
        print(sec_1)
        onesec = 0 

    if sec_1 == 10:
        # 자세 피드백 제공
        if neck >= 7:
            print("현재 목의 자세가 좋지 않습니다. 목 스트레칭을 진행해 주세요.")
        if waist >= 7:
            print("허리 자세가 좋지 않습니다. 허리 스트레칭을 권장합니다.")
        if wrist >= 7:
            print("손목 상태가 좋지 않습니다. 손목 스트레칭을 권장합니다.")
        if twist >= 7:
            print("현재 골반이 틀어질 위험이 있습니다. 골반 스트레칭을 권장합니다.")

        neck = waist = wrist = twist = sec_1 = 0

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
    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    
    return img



# def draw_poses(img, poses, point_score_threshold, skeleton=default_skeleton):
#     global sec_1
#     global neck
#     global onesec
#     global maintime
#     global waist
#     global wrist
#     global twist
#     if poses.size == 0:
#         return img
#     sec = time.time() - maintime
#     onesec = onesec + sec
#     #print(onesec)
#     img_limbs = np.copy(img)
#     for pose in poses:
#         points = pose[:, :2].astype(np.int32)
#         points_scores = pose[:, 2]
#         # Draw joints.
#         #변수 선언
#         rsh = None # right shoulder
#         lsh = None # left shoulder
        
#         rear = None # right ear
#         lear = None # left ear

#         rpvis = None # right pelvis
#         lpvis = None # lef pelvis

#         rank = None # right ankle
#         lank = None # left ankle

#         rwrist = None # right wrist
#         relbow = None # right elbow
        
#         lwrist = None # left  wrist
#         lelbow = None # left  elbow

#         for i, (p, v) in enumerate(zip(points, points_scores)):
#             if v > point_score_threshold:
#                 cv2.circle(img, tuple(p), 5, colors[i], 5)
#                 #print(tuple(p),colors[i])
#                 if i == 3:
#                     rear = tuple(p)
#                 if i == 4 :
#                     lear = tuple(p)
#                 if i == 5:
#                     rsh = tuple(p)
#                 if i == 6:
#                     lsh = tuple(p)
#                 if i == 7:
#                     relbow = tuple(p)
#                 if i == 9:
#                     rwrist = tuple(p)
#                 if i == 11:
#                     rpvis = tuple(p)
#                 if i == 12:
#                     lpvis = tuple(p)
#                 if i == 15:
#                     rank = tuple(p)
#                 if i == 16:
#                     lank = tuple(p)


#         if onesec >= 1:
#             #거북목(좌/우)    
#             if rsh is not None and rear is not None:
#                 if abs(rsh[0] - rear[0]) > 40:
#                     neck += 1
#                     #print(neck)
                    
                    

#             if lsh is not None and lear is not None:
#                 if abs(lsh[0] - lear[0]) > 40:
#                     neck += 1
#                     #print(neck)
                    
                    
#             #허리(좌/우)
#             if rsh is not None and rpvis is not None:
#                 if abs(rsh[0] - rpvis[0]) > 80:
#                     waist += 1
#                     print("waist:",waist)

#             if lsh is not None and lpvis is not None:
#                 if abs(lsh[0] - lpvis[0]) > 80:
#                     waist += 1
#                     print("waist:",waist)
                     
#             #다리 꼬기(발목 높이가 기준)
#             if rank is not None and lank is not None:
#                 if abs(rank[1] - lank[1]) > 30:
#                     twist += 1
#                     print("twist:",twist)
#             #손목 상하상태(좌/우)
#             if rwrist is not None and relbow is not None:
#                 if relbow[1] - rwrist[1]  > 50:
#                     wrist += 1
#                     print("wrist:",wrist)
            
#             if lwrist is not None and lelbow is not None:
#                 if lelbow[1] - lwrist[1]  > 50:
#                     wrist += 1
#                     print("wrist:",wrist)
#             sec_1 += 1
#             print(sec_1)
#             onesec = 0 
            
#         if sec_1 == 10:
#             if neck >= 7:
#                 print("현재 목의 자세가 좋지 않습니다. 목 스트레칭을 진행해 주세요.") 
#             if waist >= 7:
#                 print("허리 수술 3500만원")
#                 waist = 0
#             else:
#                 waist = 0

#             if wrist >= 7 :
#                 print("페이커도 손목부상 당하더라...")
#                 wrist = 0
#             else:
#                 wrist = 0

#             if twist >= 7:
#                 print("현재 골반이 틀어질 위험이 있습니다 골반 스트레칭을 권장합니다.")
#                 twist = 0
#             else:
#                 twist = 0

#             sec_1 = 0

    #     # Draw limbs.
    #     for i, j in skeleton:
    #         if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
    #             cv2.line(
    #                 img_limbs,
    #                 tuple(points[i]),
    #                 tuple(points[j]),
    #                 color=colors[j],
    #                 thickness=4,
    #             )
    # cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    
    # return img


# Main processing function to run pose estimation.
def run_pose_estimation(source=0, flip=False, use_popup=False, skip_first_frames=0):
    global maintime
    pafs_output_key = compiled_model.output("Mconv7_stage2_L1")
    heatmaps_output_key = compiled_model.output("Mconv7_stage2_L2")
    player = None
    try:
        # Create a video player to play with target fps.
        player = utils.VideoPlayer(source, flip=flip, fps=30, skip_first_frames=skip_first_frames)
        # Start capturing.
        player.start()
        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

        processing_times = collections.deque()

        while True:
            # Grab the frame.
            maintime = time.time()
            frame = player.next()
            if frame is None:
                print("Source ended")
                break
            # If the frame is larger than full HD, reduce size to improve the performance.
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # Resize the image and change dims to fit neural network input.
            # (see https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/human-pose-estimation-0001)
            input_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            # Create a batch of images (size = 1).
            input_img = input_img.transpose((2, 0, 1))[np.newaxis, ...]

            # Measure processing time.
            start_time = time.time()
            # Get results.
            results = compiled_model([input_img])
            stop_time = time.time()

            pafs = results[pafs_output_key]
            heatmaps = results[heatmaps_output_key]
            # Get poses from network results.
            poses, scores = process_results(frame, pafs, heatmaps)

            # Draw poses on a frame.
            frame = draw_poses(frame, poses, 0.1)

            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # mean processing time [ms]
            # processing_time = np.mean(processing_times) * 1000
            # fps = 1000 / processing_time
            # cv2.putText(
            #     frame,
            #     f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
            #     (20, 40),
            #     cv2.FONT_HERSHEY_COMPLEX,
            #     f_width / 1000,
            #     (0, 0, 255),
            #     1,
            #     cv2.LINE_AA,
            # )

            # Use this workaround if there is flickering.
            if use_popup:
                cv2.imshow(title, frame)
                key = cv2.waitKey(1)
                # escape = 27
                if key == 27:
                    break
            else:
                # Encode numpy array to jpg.
                _, encoded_img = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 90])
                # Create an IPython image.
                i = display.Image(data=encoded_img)
                # Display the image in this notebook.
                display.clear_output(wait=True)
                display.display(i)
    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            # Stop capturing.
            player.stop()
        if use_popup:
            cv2.destroyAllWindows()

USE_WEBCAM = True
cam_id = 0
video_file = "https://storage.openvinotoolkit.org/data/test_data/videos/store-aisle-detection.mp4"
source = cam_id if USE_WEBCAM else video_file

additional_options = {"skip_first_frames": 500} if not USE_WEBCAM else {}
run_pose_estimation(source=source, flip=isinstance(source, int), use_popup=True, **additional_options)