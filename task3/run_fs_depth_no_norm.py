import os
import sys
import cv2
import csv
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

# ================= Configuration Area =================

# 1. FoundationStereo codebase path (Please ensure this path is correct)
FS_REPO = "/vision/u/yinhang/FoundationStereo"
sys.path.append(FS_REPO)

# Attempt to import the wrapped function
try:
    from fs_wrapper import run_foundation_stereo
except ImportError:
    print(f"[ERROR] Could not import 'fs_wrapper'. Check FS_REPO path: {FS_REPO}")
    sys.exit(1)

# 2. Output root directory
OUTPUT_ROOT = "/vision/u/yinhang/pre_process/task3/depth_output_no_norm"

# ===========================================

def process_single_video(stereo_path, out_mp4_path):
    """
    Read SBS video -> Split -> Foundation Stereo Inference -> Clip(0,255) -> Save MP4
    """
    cap = cv2.VideoCapture(stereo_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {stereo_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Side-by-Side split width
    half_w = width // 2
    
    # Prepare Writer
    # Output video is monocular size (height, half_w)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_mp4_path), exist_ok=True)
    writer = cv2.VideoWriter(out_mp4_path, fourcc, fps, (half_w, height), isColor=False)
    
    # Process frame by frame
    for _ in range(total_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        # 1. BGR to RGB (Foundation Stereo requires RGB)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 2. Split left and right eyes
        left_rgb = frame_rgb[:, :half_w, :]
        right_rgb = frame_rgb[:, half_w:, :]
        
        # 3. Core: Inference
        # run_foundation_stereo returns a (H, W) numpy array (float)
        disp = run_foundation_stereo(left_rgb, right_rgb)
        
        # 4. Core: No Normalization Output
        # Clip directly to 0-255, convert to uint8
        img_8bit = np.clip(disp, 0.0, 255.0).astype(np.uint8)
        
        writer.write(img_8bit)

    cap.release()
    writer.release()
    # print(f"[SAVE] {out_mp4_path}") # Can comment out if too many logs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", type=str, required=True, help="Path to the job list TSV")
    # Sharding parameters for scaling up
    parser.add_argument("--start-idx", type=int, default=0, help="Start index")
    parser.add_argument("--end-idx", type=int, default=None, help="End index")
    args = parser.parse_args()

    # 1. Read TSV
    all_jobs = []
    if not os.path.exists(args.tsv):
        print(f"[ERROR] TSV not found: {args.tsv}")
        return

    with open(args.tsv, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            all_jobs.append(row)

    # 2. Determine processing range
    total = len(all_jobs)
    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else total
    end = min(end, total)
    
    jobs_to_process = all_jobs[start:end]
    print(f"[INFO] Processing jobs {start} to {end} (Total in batch: {len(jobs_to_process)})")

    # 3. Loop processing
    # Use tqdm to show progress bar
    for job in tqdm(jobs_to_process):
        stereo_path = job["stereo_mp4"]
        role = job["role"]     # 'wrist' or 'robot'
        lab = job["lab"]       # e.g., 'CLVR'
        uuid = job["uuid"]     # e.g., 'CLVR+...'

        # Construct output path: output_root/Lab/UUID/role_depth.mp4
        # This structure is clearest and convenient for finding data later
        save_dir = os.path.join(OUTPUT_ROOT, lab, uuid)
        out_filename = f"{role}_depth.mp4"
        out_mp4_path = os.path.join(save_dir, out_filename)

        # Simple skip mechanism (resume from checkpoint)
        if os.path.exists(out_mp4_path):
            # print(f"[SKIP] Exists: {out_mp4_path}")
            continue
            
        try:
            process_single_video(stereo_path, out_mp4_path)
        except Exception as e:
            print(f"[ERROR] Failed on {stereo_path}: {e}")

if __name__ == "__main__":
    main()