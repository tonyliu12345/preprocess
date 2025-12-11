# process_droid_caption_qwen.py

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # 避免 transformers 去 import TF

# 👇 让 TensorFlow 完全不用 GPU（避免和 Qwen 抢显存）
import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except Exception as e:
    print("tf.config.set_visible_devices([], 'GPU') failed:", e)

import tensorflow_datasets as tfds
import json
from pathlib import Path
import numpy as np
from PIL import Image

from qwen_vl_caption import qwen_vl_caption

DROID_DIR = "/viscam/data/DROID/droid/1.0.1"
OUT_DIR = Path("/vision/u/yinhang/pre_process/task2_outputs")

MIN_STEPS = 60
MAX_EPISODES = 10000   # 👈 目标 10k 轨迹


def get_first_step_and_len(steps_ds):
    it = iter(steps_ds)
    try:
        first_step = next(it)
    except StopIteration:
        return None, 0

    length = 1
    for _ in range(2000):
        try:
            _ = next(it)
            length += 1
        except StopIteration:
            break

    return first_step, length


def process_droid():
    print(f"[INFO] Loading DROID builder from: {DROID_DIR}")
    builder = tfds.builder_from_directory(DROID_DIR)
    ds = builder.as_dataset(split="train", shuffle_files=False)
    print("[INFO] Dataset loaded. Starting iteration...")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output dir: {OUT_DIR}")

    kept = 0
    total_seen = 0
    short_episodes = 0
    empty_episodes = 0
    no_instr_episodes = 0

    for epi_idx, episode in enumerate(ds):
        total_seen += 1

        steps_ds = episode["steps"]
        first_step, length = get_first_step_and_len(steps_ds)

        if first_step is None:
            empty_episodes += 1
            if total_seen % 500 == 0:
                print(f"[SKIP] tfds_idx={epi_idx}: empty episode (empty so far: {empty_episodes})")
            continue

        if length < MIN_STEPS:
            short_episodes += 1
            if total_seen % 500 == 0:
                print(
                    f"[SKIP] tfds_idx={epi_idx}: too short (len={length} < {MIN_STEPS}), "
                    f"short_episodes={short_episodes}"
                )
            continue

        obs = first_step["observation"]
        img_np = obs["exterior_image_1_left"].numpy()  # (H,W,3)

        # 1 canonical instruction
        instruction = first_step["language_instruction"].numpy().decode("utf-8")
        instr_clean = instruction.strip()
        if not instr_clean:
            no_instr_episodes += 1

        # Call Qwen-VL (核心耗时部分)
        # TODO - ADD INTRUCTION TO THE SCENE_CAPTION PROMPT & MENTION "The scene is: ..." BEFORE CAPTION
        try:
            scene_caption = qwen_vl_caption(img_np)
        except Exception as e:
            # 防止单条炸掉整个 job
            print(f"[ERROR] Qwen caption failed at tfds_idx={epi_idx}, kept={kept}: {e}")
            continue

        episode_id = f"episode_{kept:06d}"
        img_path = OUT_DIR / f"{episode_id}.png"
        Image.fromarray(img_np).save(img_path)

        # 拼 combined 字段
        if instr_clean:
            combined = (
                scene_caption.rstrip(". ") + ". "
                + f"The robot is instructed to: {instr_clean}"
            )
        else:
            combined = scene_caption

        record = {
            "episode_id": episode_id,
            "instruction": instruction,
            "scene_caption": scene_caption,
            "combined_description": combined,
            "image_path": str(img_path),
            "num_steps": int(length),
        }

        json_path = OUT_DIR / f"{episode_id}.json"
        with open(json_path, "w") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        # 进度 log
        print(
            f"[{kept}] saved {episode_id}.json & {episode_id}.png | "
            f"tfds_idx={epi_idx}, len={length}, "
            f"instr_empty={not bool(instr_clean)}"
        )

        kept += 1
        if kept >= MAX_EPISODES:
            print(f"[INFO] Reached MAX_EPISODES={MAX_EPISODES}, stopping.")
            break

    # 结束时打印 summary
    print("\n===== SUMMARY =====")
    print(f"Total TFDS episodes seen: {total_seen}")
    print(f"Empty episodes         : {empty_episodes}")
    print(f"Too-short episodes (<{MIN_STEPS}) : {short_episodes}")
    print(f"Episodes with empty instruction    : {no_instr_episodes}")
    print(f"Episodes kept (written)            : {kept}")
    print(f"Output dir: {OUT_DIR}")
    print("===================\n")


if __name__ == "__main__":
    process_droid()
