### **README.md**

````
# Task 1: Robot RGB Mask Generation (SAM3)

This directory contains the scripts for **Step 1** of the data processing pipeline: generating robot segmentation masks for the DROID dataset using the SAM3 model.

The goal is to process raw RGB frames from the DROID RLDS dataset, segment the "robot arm" using text prompts, and save the resulting binary masks as MP4 videos.

## Files

* **`droid_sam3_multi.py`**:
    The core Python script that:
    * Loads the DROID dataset (RLDS format) from `/viscam/data/DROID/droid/1.0.1`.
    * Disables TensorFlow GPU usage to reserve VRAM for PyTorch/SAM3.
    * Runs SAM3 inference on specific camera views (`exterior_image_1_left`, `exterior_image_2_left`).
    * Saves the original RGB video and the generated Mask video.
    *

* **`sam3_droid_array.sbatch.sh`**:
    The Slurm job submission script that:
    * Distributes the processing across 32 GPU nodes using a strided array job pattern.
    * Manages offsets and counts to process a target of 10,000 episodes in parallel chunks.
    *

## Environment Setup

Ensure you have the `sam3` conda environment activated and access to the SVL cluster storage.

```bash
source /vision/u/yinhang/miniconda3/etc/profile.d/conda.sh
conda activate sam3
````

## Usage

### 1\. Local / Debug Run

To test the pipeline on a small number of episodes (e.g., 1 episode) without submitting a job:

```bash
# Processes 1 episode starting from index 0
python droid_sam3_multi.py --offset 0 --count 1
```

### 2\. Cluster Batch Processing (Slurm)

To process the large-scale dataset (default target: 10k episodes), submit the Slurm array script.

```bash
sbatch sam3_droid_array.sbatch.sh
```

**How the Batching Works:**
The script utilizes a strided index approach to avoid timeouts and maximize parallelism:

  * **Array Size:** 32 tasks (`0-31`).
  * **Chunk Size:** 20 episodes per Python execution.
  * **Logic:** Task $N$ processes a chunk, then skips ahead by $32 \times 20$ episodes, continuing until the `TOTAL_EPISODES` limit is reached.

## Configuration

### Dataset & Paths

  * **Input Data:** `/viscam/data/DROID/droid/1.0.1` (RLDS format).
  * **Output Root:** `outputs/sam3`.

### Camera & Prompts

Currently configured to process the following cameras with the prompt **"robot arm"**:

1.  `exterior_image_1_left` (Threshold: 0.5)
2.  `exterior_image_2_left` (Threshold: 0.5)

*Note: The `wrist_image_left` is currently commented out in the configuration.*

## Output Structure

The script generates a directory structure under `outputs/sam3/droid_101/`:

```text
outputs/sam3/droid_101/
└── droid_101_ep_000000/           # Episode ID
    ├── meta.json                  # Contains language instruction and file paths
    ├── exterior_image_1_left/
    │   ├── rgb.mp4                # Original RGB Frames
    │   └── robot_mask.mp4         # Generated SAM3 Mask
    └── exterior_image_2_left/
        ├── rgb.mp4
        └── robot_mask.mp4
```

```
