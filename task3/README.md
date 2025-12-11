### **README.md**

# Task 3: Stereo Depth Video Generation

This directory contains the scripts for **Step 3** of the data processing pipeline: converting stereo RGB videos (Side-by-Side) from the DROID dataset into depth map videos using the **Foundation Stereo** model.

The pipeline selects a balanced subset of 10,000 "success" trajectories and processes them in parallel batches to generate 8-bit unnormalized depth videos.

## Files

* **`prepare_stereo_job_list.py`**:
    The data selection script that:
    * Scans the raw DROID dataset (`/viscam/data/DROID/droid_raw_t/1.0.1`).
    * **Filters:** Selects only successful trajectories (`success` folder only).
    * **Sampling:** Uses a Round-Robin approach to sample uniformly across different Labs (excluding specific labs like IPRL, RAIL, etc.) to avoid bias.
    * **Output:** Generates a TSV file listing all wrist and robot stereo video paths to process.

* **`run_fs_depth_no_norm.py`**:
    The inference script that:
    * Loads the **Foundation Stereo** model from `/vision/u/yinhang/FoundationStereo`.
    * **Preprocessing:** Splits side-by-side (SBS) stereo frames into Left/Right pairs.
    * **Inference:** Computes the disparity/depth map.
    * **Postprocessing:** Clips values to [0, 255] (without normalization) and saves as an 8-bit grayscale MP4 video.

* **`depth_video_droid.sbatch.sh`**:
    The Slurm job submission script that:
    * Distributes the workload using a Slurm Array.
    * Splits the 10,000-job TSV list into chunks (logic based on `NUM_ARRAYS=30`) for parallel processing on GPU nodes.

## Environment Setup

This task requires the `FoundationStereo` repository and a specific environment.

**Dependencies:**
* Path to FoundationStereo: `/vision/u/yinhang/FoundationStereo`.
* OpenCV (`cv2`), `tqdm`, `numpy`.

## Usage

### 1. Generate Job List
First, create the list of videos to process. This scans the dataset and selects 10,000 trajectories.

```bash
# See prepare_stereo_job_list_command.txt for full command
python prepare_stereo_job_list.py \
    --dataset-root /viscam/data/DROID/droid_raw_t/1.0.1 \
    --output /vision/u/yinhang/pre_process/task3/stereo_job_list_output/fs_stereo_jobs_10000.tsv \
    --max-jobs 10000
````

### 2\. Run Depth Generation (Slurm)

Submit the array job to process the generated TSV list.

```bash
sbatch depth_video_droid.sbatch.sh
```

**Job Configuration:**

  * **Total Data:** 10,000 videos.
  * **Parallelism:** Configured to split into chunks based on `NUM_ARRAYS=30` (approx 333 videos per job).
  * **Resources:** 1 Titan RTX GPU per task.

## Configuration

### Input & Output Paths

  * **Input Data:** `/viscam/data/DROID/droid_raw_t/1.0.1` (Raw Stereo MP4s).
  * **Job List TSV:** `/vision/u/yinhang/pre_process/task3/stereo_job_list_output/fs_stereo_jobs_10000.tsv`.
  * **Output Root:** `/vision/u/yinhang/pre_process/task3/depth_output_no_norm`.

### Output Structure

The output preserves the Lab/UUID structure of the original dataset:

```text
/vision/u/yinhang/pre_process/task3/depth_output_no_norm/
└── {Lab_Name}/
    └── {UUID}/
        ├── wrist_depth.mp4    # Depth from wrist stereo camera
        └── robot_depth.mp4    # Depth from robot exterior stereo camera
```

*Note: The script implements a skip mechanism; if the output file already exists, it will be skipped during processing.*

```
```