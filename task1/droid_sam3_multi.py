import os
from pathlib import Path
import json
import argparse

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ===== 基本配置 =====
# DATASET_NAME = "droid_100"
# DATA_DIR = "gs://gresearch/robotics"
# OUT_ROOT = Path("outputs/sam3")  # 所有 SAM3 结果放这里

# ===== 配置 =====
DROID_DIR = "/viscam/data/DROID/droid/1.0.1"   # ✅ 你刚刚验证过的路径
DATASET_NAME = "droid_101"                     # 来自 builder.info.name
OUT_ROOT = Path("outputs/sam3")


# 想处理多少条 episode 由命令行 --count 决定，这里不再写死

# 现在先只跑两个 exterior 视角；wrist 后面单独研究
CAMERA_KEYS = [
    "exterior_image_1_left",
    "exterior_image_2_left",
    # "wrist_image_left",
]

# 默认文本 prompt（camera-specific 在下面 CAMERA_PROMPTS 里）
TEXT_PROMPT = "robot arm"

# camera 专属 prompt
CAMERA_PROMPTS = {
    "exterior_image_1_left": "robot arm",
    "exterior_image_2_left": "robot arm",
    # "wrist_image_left": "robot gripper at the bottom of the image",
}

# 每个 camera 的二值化阈值
CAMERA_THRESHOLDS = {
    "exterior_image_1_left": 0.5,
    "exterior_image_2_left": 0.5,
    # "wrist_image_left": 0.05,
}

# ===== 关掉 TensorFlow GPU，只让 TF 用 CPU（避免跟 PyTorch 抢显存）=====
try:
    tf.config.set_visible_devices([], "GPU")
    print("Disabled TensorFlow GPU; TF will run on CPU.")
except Exception as e:
    print("Could not disable TF GPU:", e)
# =====================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("PyTorch device:", DEVICE)


# ===== SAM3 相关 =====
def load_sam3_model():
    print(f"Loading SAM3 image model on {DEVICE} ...")
    model = build_sam3_image_model()
    model.to(DEVICE)
    processor = Sam3Processor(model)
    return model, processor


def run_sam3_on_frame(model, processor, frame_rgb: np.ndarray,
                      prompt: str, threshold: float) -> np.ndarray:
    """
    输入: frame_rgb: (H, W, 3) uint8, RGB
    输出: mask: (H, W) uint8, 0 或 255
    """
    pil_img = Image.fromarray(frame_rgb)
    state = processor.set_image(pil_img)
    output = processor.set_text_prompt(state=state, prompt=prompt)

    masks = output["masks"]  # 可能是 [B, N, H, W] / [N, H, W] / [H, W]

    if isinstance(masks, torch.Tensor):
        masks_np = masks.detach().cpu().numpy()
    else:
        masks_np = np.asarray(masks)

    # 没任何 mask：直接全黑
    if masks_np.size == 0:
        H, W = frame_rgb.shape[:2]
        return np.zeros((H, W), dtype=np.uint8)

    # 标准化 shape
    if masks_np.ndim == 4:
        # (B, N, H, W) -> (K, H, W)
        B, N, H, W = masks_np.shape
        masks_np = masks_np.reshape(B * N, H, W)
    elif masks_np.ndim == 3:
        # (N, H, W) 或 (H, W, 1) -> 暂时不动
        pass
    elif masks_np.ndim == 2:
        mask = (masks_np > threshold).astype(np.uint8) * 255
        return mask
    else:
        raise ValueError(f"Unsupported masks ndim={masks_np.ndim}, shape={masks_np.shape}")

    # 到这里期望 (K, H, W)
    if masks_np.ndim == 3 and masks_np.shape[0] == 1:
        union = masks_np[0]
    else:
        union = masks_np.max(axis=0)  # (H, W)

    max_score = float(union.max())

    if max_score < threshold:
        # 兜底策略：所有分数都很低，直接选平均激活最高的那一张 mask
        flat = masks_np.reshape(masks_np.shape[0], -1)
        best_idx = int(flat.mean(axis=1).argmax())
        best_mask = masks_np[best_idx]
        # 对这个 best_mask 用更宽松的二值化
        mask = (best_mask > 0.10).astype(np.uint8) * 255
    else:
        mask = (union > threshold).astype(np.uint8) * 255

    return mask


# ===== 一些小工具函数 =====
def get_episode_id(global_idx: int) -> str:
    """
    简单稳定：用 global index，保证唯一性 & 可排序。
    """
    return f"{DATASET_NAME}_ep_{global_idx:06d}"


def extract_instruction(episode) -> str:
    """
    从第一步里抽 language_instruction
    """
    steps = episode["steps"]
    for step in steps:
        return step["language_instruction"].numpy().decode("utf-8")
    return ""


def extract_frames_per_camera(episode, camera_keys):
    """
    把一个 episode 里的多 camera frame 都变成 numpy:
    返回:
      imgs_per_cam: dict[camera_key] -> np.ndarray (T, H, W, 3)
    """
    steps = episode["steps"]

    frames_per_cam = {cam: [] for cam in camera_keys}

    for step in steps:
        obs = step["observation"]
        for cam in camera_keys:
            if cam in obs:
                frames_per_cam[cam].append(obs[cam].numpy())

    imgs_per_cam = {}
    for cam, frames in frames_per_cam.items():
        if len(frames) == 0:
            print(f"[WARN] No frames found for camera '{cam}' in this episode, skip.")
            continue
        imgs_per_cam[cam] = np.stack(frames, axis=0)  # (T, H, W, 3)

    return imgs_per_cam


def write_videos_for_camera(
    cam_dir: Path,
    imgs: np.ndarray,
    model,
    processor,
    prompt: str,
    threshold: float,
) -> dict:
    """
    写一个相机的：
      - rgb.mp4
      - robot_mask.mp4
    返回一个 meta dict
    """
    cam_dir.mkdir(parents=True, exist_ok=True)
    T, H, W, _ = imgs.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # RGB
    rgb_path = cam_dir / "rgb.mp4"
    rgb_writer = cv2.VideoWriter(str(rgb_path), fourcc, 15, (W, H))
    for t in range(T):
        rgb_writer.write(imgs[t][..., ::-1])  # RGB -> BGR
    rgb_writer.release()

    # MASK
    mask_path = cam_dir / "robot_mask.mp4"
    mask_writer = cv2.VideoWriter(str(mask_path), fourcc, 15, (W, H), isColor=True)

    for t in range(T):
        frame = imgs[t]
        mask = run_sam3_on_frame(model, processor, frame, prompt, threshold)

        if mask.dtype != np.uint8:
            mask = mask.astype(np.float32)
            if mask.max() <= 1.0:
                mask = (mask * 255.0).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)

        if mask.ndim == 2:
            mask_rgb = np.repeat(mask[..., None], 3, axis=-1)
        elif mask.ndim == 3 and mask.shape[2] == 1:
            mask_rgb = np.repeat(mask, 3, axis=-1)
        elif mask.ndim == 3 and mask.shape[2] == 3:
            mask_rgb = mask
        else:
            raise ValueError(f"Unexpected mask shape {mask.shape}")

        mask_writer.write(mask_rgb)

    mask_writer.release()

    return {
        "num_frames": int(T),
        "rgb_video": rgb_path.name,
        "mask_video": mask_path.name,
    }


# ===== 处理单个 episode =====
def process_one_episode(episode, global_idx: int, dataset_out_dir: Path,
                        model, processor):
    episode_id = get_episode_id(global_idx)
    ep_dir = dataset_out_dir / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Episode {global_idx} -> id={episode_id} ===")

    instr = extract_instruction(episode)
    print("Instruction:", instr)

    # 拿到每个相机的 numpy 帧
    imgs_per_cam = extract_frames_per_camera(episode, CAMERA_KEYS)

    cameras_meta = {}

    for cam, imgs in imgs_per_cam.items():
        prompt = CAMERA_PROMPTS.get(cam, TEXT_PROMPT)
        thr = CAMERA_THRESHOLDS.get(cam, 0.5)
        cam_dir = ep_dir / cam
        cam_meta = write_videos_for_camera(cam_dir, imgs, model, processor, prompt, thr)
        cameras_meta[cam] = {
            "num_frames": cam_meta["num_frames"],
            "rgb_video": f"{cam}/{cam_meta['rgb_video']}",
            "mask_video": f"{cam}/{cam_meta['mask_video']}",
        }

    # 写 episode 级别 meta.json
    meta = {
        "dataset": DATASET_NAME,
        "episode_id": episode_id,
        "language_instruction": instr,
        "cameras": cameras_meta,
    }

    meta_path = ep_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved meta to: {meta_path}")

# ===== 主流程：支持 offset / count，用于 Slurm array 切分 =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, default=0,
                        help="跳过前多少个 episode（global index）")
    parser.add_argument("--count", type=int, default=1,
                        help="本次处理多少个 episode")
    args = parser.parse_args()

    dataset_out_dir = OUT_ROOT / DATASET_NAME
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    # ✅ 用本地 RLDS droid_101，而不是 gs:// 的 droid_100
    print(f"Loading builder from {DROID_DIR} ...")
    builder = tfds.builder_from_directory(DROID_DIR)
    info = builder.info
    print(
        f"Loaded dataset: {info.name}, "
        f"train num_examples={info.splits['train'].num_examples}, "
        f"num_shards={info.splits['train'].num_shards}"
    )

    ds = builder.as_dataset(
        split="train",
        read_config=tfds.ReadConfig(try_autocache=False),
    )

    # 用 offset / count 截取一段（方便 Slurm array 并行）
    ds = ds.skip(args.offset).take(args.count)

    # 整个脚本只 load 一次 SAM3
    model, processor = load_sam3_model()

    for local_idx, episode in enumerate(ds):
        global_idx = args.offset + local_idx
        process_one_episode(episode, global_idx, dataset_out_dir, model, processor)

    print("\nAll done!")



if __name__ == "__main__":
    main()
