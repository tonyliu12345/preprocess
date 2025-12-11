### **README.md**

# DROID Data Pre-processing Pipeline

This project implements a multi-stage data processing pipeline for the DROID dataset. It transforms raw RLDS episodes and stereo videos into semantic masks, natural language descriptions, and depth maps for downstream policy training.

The pipeline is divided into three distinct tasks, each handling a specific modality.

## 📁 Directory Structure & Outputs

| Task | Modality | Output Directory | Environment |
| :--- | :--- | :--- | :--- |
| **Task 1** | Robot Segmentation | `/vision/u/yinhang/pre_process/task1/outputs/sam3/droid_101` | `sam3` |
| **Task 2** | Image Captioning | `/vision/u/yinhang/pre_process/task2_outputs` | `sam3` |
| **Task 3** | Stereo Depth | `/vision/u/yinhang/pre_process/task3/depth_output_no_norm` | `fs3` |

---

## 🚀 Pipeline Overview

### Task 1: Robot RGB Mask Generation (SAM3)
Generates binary segmentation masks for the robot arm using the **SAM3** model. It processes RGB frames from specific camera views and outputs masked video files.

* **Code Behavior:** Currently configured to process `exterior_image_1_left` and `exterior_image_2_left` with the prompt "robot arm".
* **Note:** Wrist camera processing (`wrist_image_left`) is currently **disabled** in the configuration.
* **Environment:** `conda activate sam3`
* **More Info:** See [task1/README.md](task1/README.md)

### Task 2: Scene Captioning & Instruction Alignment (Qwen-VL)
Generates semantic scene descriptions using **Qwen2-VL-7B-Instruct**. It combines visual captions of the first frame with the original robot task instructions.

* **Code Behavior:** Filters for episodes >60 steps, prompts the VLM for concise object descriptions (excluding background/furniture), and merges this with the task instruction into a `combined_description`.
* **Environment:** `conda activate sam3`
* **More Info:** See [task2/README.md](task2/README.md)

### Task 3: Stereo Depth Generation (Foundation Stereo)
Converts raw stereo (Side-by-Side) MP4 videos into unnormalized depth maps using the **Foundation Stereo** model.

* **Code Behavior:** Splits stereo frames into left/right pairs, computes disparity, clips values to [0, 255], and saves as 8-bit grayscale MP4s. It targets a balanced subset of 10,000 successful trajectories.
* **Environment:** `conda activate fs3`
* **More Info:** See [task3/README.md](task3/README.md)

---

## 🛠️ Usage Summary

### 1. Run Task 1 (Masks)
```bash
conda activate sam3
cd task1
# Submit array job
sbatch sam3_droid_array.sbatch.sh
````

### 2\. Run Task 2 (Captions)

```bash
conda activate sam3
cd task2
# Submit single job
sbatch droid_qwen7b.sbatch.sh
```

### 3\. Run Task 3 (Depth)

```bash
conda activate fs3
cd task3
# Generate job list first, then submit array
python prepare_stereo_job_list.py ...
sbatch depth_video_droid.sbatch.sh
```

```
```