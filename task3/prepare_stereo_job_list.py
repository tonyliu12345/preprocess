#!/usr/bin/env python3
"""
prepare_stereo_job_list_rr_success.py

Modified version based on Round-Robin:
1. **Only scan success directories**: Completely ignore failure data.
2. Maintain original features like uniform sampling, speed, and skipping specific Labs.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Iterator

import cv2

EXCLUDED_LABS = {"IPRL", "RAIL", "RAD", "WEIRD"}

# Simple reject statistics
REJECT_STATS = {
    "total_scanned": 0,
    "valid_found": 0,
    "reject_details": {}
}

def log_reject(reason: str):
    REJECT_STATS["reject_details"][reason] = REJECT_STATS["reject_details"].get(reason, 0) + 1

# ----------------- Basic Utility Functions ----------------- #

def resolve_stereo_mp4(root_dir: Path, rel_mp4: str) -> Tuple[Optional[Path], bool]:
    rel_path = Path(rel_mp4)
    # 1. Check directly
    if rel_path.stem.endswith("-stereo"):
        p = (root_dir / rel_path).resolve()
        return p, p.is_file()
    # 2. Construct -stereo
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
    """Attempt to build trajectory, return None if failed"""
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

# ----------------- Core Logic: Lab Round-Robin (Modified) ----------------- #

def get_metadata_iterator(lab_root: Path) -> Iterator[Path]:
    """Generate a file iterator for a Lab, restricted to scanning only the success subdirectory"""
    success_dir = lab_root / "success"
    
    # If the Lab has no success directory, return directly (generator ends)
    if not success_dir.is_dir():
        return

    for root, _, files in os.walk(str(success_dir)):
        for name in files:
            if name.startswith("metadata_") and name.endswith(".json"):
                yield Path(root) / name

def run_round_robin(dataset_root: Path, max_trajs: int, min_dur: float) -> List[Dict]:
    # 1. Find all valid Lab directories
    all_subdirs = [d for d in dataset_root.iterdir() if d.is_dir()]
    valid_labs = []
    
    for d in all_subdirs:
        lab_name = d.name
        if lab_name in EXCLUDED_LABS:
            continue
        valid_labs.append(lab_name)
    
    random.shuffle(valid_labs)
    print(f"[INFO] Found {len(valid_labs)} labs: {valid_labs}")

    # 2. Initialize an iterator for each Lab
    lab_iterators = {}
    for lab in valid_labs:
        lab_root = dataset_root / lab
        lab_iterators[lab] = get_metadata_iterator(lab_root)

    jobs = []
    collected_uuids = set()
    
    # 3. Round-Robin
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
                        break # This round complete, exit inner loop
                
                # Prevent taking too long in a single pass
                if scan_attempt > 50:
                    break
            
            # Put back to the end of the queue
            active_labs.append(current_lab)

        except StopIteration:
            # The success directory of this Lab is exhausted
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
    # Compatibility arguments
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