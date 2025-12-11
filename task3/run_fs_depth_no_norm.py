import os
import sys
import cv2
import csv
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================

# 1. FoundationStereo 代码库路径 (请确保这个路径是对的)
FS_REPO = "/vision/u/yinhang/FoundationStereo"
sys.path.append(FS_REPO)

# 尝试导入封装好的函数
try:
    from fs_wrapper import run_foundation_stereo
except ImportError:
    print(f"[ERROR] Could not import 'fs_wrapper'. Check FS_REPO path: {FS_REPO}")
    sys.exit(1)

# 2. 输出结果的根目录
OUTPUT_ROOT = "/vision/u/yinhang/pre_process/task3/depth_output_no_norm"

# ===========================================

def process_single_video(stereo_path, out_mp4_path):
    """
    读取 SBS 视频 -> 拆分 -> Foundation Stereo 推理 -> Clip(0,255) -> 保存 MP4
    """
    cap = cv2.VideoCapture(stereo_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {stereo_path}")
        return

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Side-by-Side 拆分宽度
    half_w = width // 2
    
    # 准备 Writer
    # 输出视频是单眼尺寸 (height, half_w)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_mp4_path), exist_ok=True)
    writer = cv2.VideoWriter(out_mp4_path, fourcc, fps, (half_w, height), isColor=False)
    
    # 逐帧处理
    for _ in range(total_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        # 1. BGR 转 RGB (Foundation Stereo 需要 RGB)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 2. 拆分左右眼
        left_rgb = frame_rgb[:, :half_w, :]
        right_rgb = frame_rgb[:, half_w:, :]
        
        # 3. 核心：推理
        # run_foundation_stereo 返回的是 (H, W) 的 numpy array (float)
        disp = run_foundation_stereo(left_rgb, right_rgb)
        
        # 4. 核心：No Normalization Output
        # 直接截断到 0-255，转 uint8
        img_8bit = np.clip(disp, 0.0, 255.0).astype(np.uint8)
        
        writer.write(img_8bit)

    cap.release()
    writer.release()
    # print(f"[SAVE] {out_mp4_path}") # 太多 log 可以注释掉

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", type=str, required=True, help="Path to the job list TSV")
    # 用于 scale up 的分片参数
    parser.add_argument("--start-idx", type=int, default=0, help="Start index")
    parser.add_argument("--end-idx", type=int, default=None, help="End index")
    args = parser.parse_args()

    # 1. 读取 TSV
    all_jobs = []
    if not os.path.exists(args.tsv):
        print(f"[ERROR] TSV not found: {args.tsv}")
        return

    with open(args.tsv, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            all_jobs.append(row)

    # 2. 确定处理范围
    total = len(all_jobs)
    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else total
    end = min(end, total)
    
    jobs_to_process = all_jobs[start:end]
    print(f"[INFO] Processing jobs {start} to {end} (Total in batch: {len(jobs_to_process)})")

    # 3. 循环处理
    # 使用 tqdm 显示进度条
    for job in tqdm(jobs_to_process):
        stereo_path = job["stereo_mp4"]
        role = job["role"]     # 'wrist' or 'robot'
        lab = job["lab"]       # e.g., 'CLVR'
        uuid = job["uuid"]     # e.g., 'CLVR+...'

        # 构造输出路径: output_root/Lab/UUID/role_depth.mp4
        # 这种结构最清晰，方便以后找数据
        save_dir = os.path.join(OUTPUT_ROOT, lab, uuid)
        out_filename = f"{role}_depth.mp4"
        out_mp4_path = os.path.join(save_dir, out_filename)

        # 简单的跳过机制 (断点续传)
        if os.path.exists(out_mp4_path):
            # print(f"[SKIP] Exists: {out_mp4_path}")
            continue
            
        try:
            process_single_video(stereo_path, out_mp4_path)
        except Exception as e:
            print(f"[ERROR] Failed on {stereo_path}: {e}")

if __name__ == "__main__":
    main()