#!/usr/bin/env python3
"""
prepare_stereo_job_list_rr_success.py

基于 Round-Robin 版本的修改版：
1. **只扫描 success 目录**：完全忽略 failure 数据。
2. 保持原有的均匀采样、快速、跳过指定 Lab 等特性。
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Iterator

import cv2

EXCLUDED_LABS = {"IPRL", "RAIL", "RAD", "WEIRD"}

# 简单的 reject 统计
REJECT_STATS = {
    "total_scanned": 0,
    "valid_found": 0,
    "reject_details": {}
}

def log_reject(reason: str):
    REJECT_STATS["reject_details"][reason] = REJECT_STATS["reject_details"].get(reason, 0) + 1

# ----------------- 基础工具函数 ----------------- #

def resolve_stereo_mp4(root_dir: Path, rel_mp4: str) -> Tuple[Optional[Path], bool]:
    rel_path = Path(rel_mp4)
    # 1. 直接检查
    if rel_path.stem.endswith("-stereo"):
        p = (root_dir / rel_path).resolve()
        return p, p.is_file()
    # 2. 拼凑 -stereo
    stereo_name = f"{rel_path.stem}-stereo{rel_path.suffix}"
    p = (root_dir / rel_path.with_name(stereo_name)).resolve()
    return p, p.is_file()

def video_duration_sec(path: Path) -> float:
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened(): return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        return float(cnt / fps) if fps > 0 else 0.0
    except:
        return 0.0

def build_traj(meta: Dict, lab: str, lab_root: Path, min_dur: float) -> Optional[Dict]:
    """尝试构建 trajectory，失败返回 None"""
    w_rel = meta.get("wrist_mp4_path")
    if not w_rel:
        log_reject("no_wrist_key")
        return None
    
    w_path, w_exists = resolve_stereo_mp4(lab_root, w_rel)
    if not w_exists:
        log_reject(f"wrist_missing_in_{lab}")
        return None

    # Robot: try ext1 then ext2
    r_path = None
    if meta.get("ext1_mp4_path"):
        p, exists = resolve_stereo_mp4(lab_root, meta["ext1_mp4_path"])
        if exists: r_path = p
    
    if not r_path and meta.get("ext2_mp4_path"):
        p, exists = resolve_stereo_mp4(lab_root, meta["ext2_mp4_path"])
        if exists: r_path = p
        
    if not r_path:
        log_reject(f"robot_missing_in_{lab}")
        return None

    # Check durations
    if video_duration_sec(w_path) < min_dur or video_duration_sec(r_path) < min_dur:
        log_reject("duration_short")
        return None

    return {
        "lab": meta.get("lab", lab),
        "uuid": meta.get("uuid", "unknown"),
        "success": bool(meta.get("success", True)),
        "wrist_stereo": w_path,
        "robot_stereo": r_path,
    }

def traj_to_jobs(traj: Dict) -> List[Dict]:
    base = {"lab": traj["lab"], "uuid": traj["uuid"], "success": traj["success"]}
    return [
        {**base, "stereo_mp4": str(traj["wrist_stereo"]), "role": "wrist"},
        {**base, "stereo_mp4": str(traj["robot_stereo"]), "role": "robot"},
    ]

# ----------------- 核心逻辑：Lab 轮询器 (修改版) ----------------- #

def get_metadata_iterator(lab_root: Path) -> Iterator[Path]:
    """为一个 Lab 生成一个文件遍历器，限定只扫描 success 子目录"""
    success_dir = lab_root / "success"
    
    # 如果该 Lab 没有 success 目录，直接返回（生成器结束）
    if not success_dir.is_dir():
        return

    for root, _, files in os.walk(str(success_dir)):
        for name in files:
            if name.startswith("metadata_") and name.endswith(".json"):
                yield Path(root) / name

def run_round_robin(dataset_root: Path, max_trajs: int, min_dur: float) -> List[Dict]:
    # 1. 找到所有合法的 Lab 目录
    all_subdirs = [d for d in dataset_root.iterdir() if d.is_dir()]
    valid_labs = []
    
    for d in all_subdirs:
        lab_name = d.name
        if lab_name in EXCLUDED_LABS:
            continue
        valid_labs.append(lab_name)
    
    random.shuffle(valid_labs)
    print(f"[INFO] Found {len(valid_labs)} labs: {valid_labs}")

    # 2. 为每个 Lab 初始化一个迭代器
    lab_iterators = {}
    for lab in valid_labs:
        lab_root = dataset_root / lab
        lab_iterators[lab] = get_metadata_iterator(lab_root)

    jobs = []
    collected_uuids = set()
    
    # 3. 轮询
    active_labs = list(valid_labs) 
    
    print(f"[INFO] Starting Round-Robin (SUCCESS ONLY) sampling for {max_trajs} trajectories...")

    while len(collected_uuids) < max_trajs and active_labs:
        current_lab = active_labs.pop(0)
        iterator = lab_iterators[current_lab]
        
        try:
            scan_attempt = 0
            while True:
                meta_path = next(iterator)
                REJECT_STATS["total_scanned"] += 1
                scan_attempt += 1
                
                try:
                    with meta_path.open("r") as f: meta = json.load(f)
                    traj = build_traj(meta, current_lab, dataset_root/current_lab, min_dur)
                except Exception:
                    traj = None
                
                if traj:
                    if traj["uuid"] not in collected_uuids:
                        jobs.extend(traj_to_jobs(traj))
                        collected_uuids.add(traj["uuid"])
                        REJECT_STATS["valid_found"] += 1
                        print(f"[HIT #{len(collected_uuids)}] From {current_lab} (Success): {traj['uuid']}")
                        break # 这一轮完成，退出内层循环
                
                # 防止单次占用太久
                if scan_attempt > 50:
                    break
            
            # 放回队尾
            active_labs.append(current_lab)

        except StopIteration:
            # 该 Lab 的 success 目录被掏空了
            pass
            
    return jobs

# ----------------- Main ----------------- #

def save_jobs_tsv(jobs: List[Dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write("stereo_mp4\trole\tlab\tuuid\tsuccess\n")
        for j in jobs:
            f.write(f"{j['stereo_mp4']}\t{j['role']}\t{j['lab']}\t{j['uuid']}\t{'1' if j['success'] else '0'}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-jobs", type=int, default=3)
    parser.add_argument("--min-duration-sec", type=float, default=2.0)
    # 兼容参数
    parser.add_argument("--uniform", action="store_true")
    parser.add_argument("--max-scan-per-lab", type=int, default=50)

    args = parser.parse_args()
    
    root = Path(args.dataset_root)
    
    jobs = run_round_robin(root, args.max_jobs, args.min_duration_sec)
    
    print(f"[INFO] Total Scanned: {REJECT_STATS['total_scanned']}")
    print(f"[INFO] Writing {len(jobs)} jobs to {args.output}")
    save_jobs_tsv(jobs, Path(args.output))

if __name__ == "__main__":
    main()