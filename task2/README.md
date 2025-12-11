### **README.md**

# Task 2: DROID Scene Captioning & Instruction Alignment

This directory contains the scripts for **Step 2** of the data processing pipeline. This step generates semantic scene descriptions for DROID episodes using **Qwen2-VL-7B-Instruct** and combines them with the original robot task instructions.

The pipeline processes the **first frame** of valid episodes to create a "combined description" (Scene Context + Task Instruction) intended for downstream training.

## Files

* **`process_droid_caption_qwen.py`**:
    The main driver script that:
    * Iterates through the DROID dataset (RLDS format) located at `/viscam/data/DROID/droid/1.0.1`.
    * **Filters Data:** Skips empty episodes or trajectories shorter than **60 steps**.
    * **Extracts:** The first RGB frame (`exterior_image_1_left`) and the natural language instruction.
    * **Combines:** Merges the generated visual caption with the instruction into a single `combined_description` field.
    * **Output:** Saves the reference image (PNG) and metadata (JSON) for up to 10,000 episodes.

* **`qwen_vl_caption.py`**:
    The model inference module that:
    * Loads `Qwen/Qwen2-VL-7B-Instruct` with 4-bit/float16 optimizations.
    * **Prompting:** Uses a specifically tuned prompt to describe *only* visible tabletop objects, colors, and positions, while explicitly ignoring background furniture, the robot arm, and "assistant" role-play text.
    * **Cleaning:** Post-processes the model output to strip prompt echoes and "assistant:" headers.

* **`droid_qwen7b.sbatch.sh`**:
    The Slurm job submission script that:
    * Requests a single Titan RTX node (1 GPU, 128GB RAM) for 5 days.
    * Runs the `process_droid_caption_qwen.py` script.
    * *Note: Unlike Task 1, this is a single job, not an array, as the script handles the dataset iteration internally.*

## Environment Setup

The scripts require the `transformers` library and `qwen_vl_utils`. It is recommended to use the provided conda environment (referenced as `sam3` in the scripts, though it must support Qwen-VL).

```bash
source /vision/u/yinhang/miniconda3/etc/profile.d/conda.sh
conda activate sam3
````

*Note: The scripts automatically disable TensorFlow GPU visibility (`tf.config.set_visible_devices([], "GPU")`) to prevent memory conflicts with PyTorch/Qwen.*

## Usage

### 1\. Local / Debug Run

To run the processing script interactively (useful for debugging prompt outputs):

```bash
python process_droid_caption_qwen.py
```

### 2\. Cluster Processing (Slurm)

To process the full dataset (target: 10k episodes), submit the batch job:

```bash
sbatch droid_qwen7b.sbatch.sh
```

## Configuration

### Data Filtering

The script currently enforces strict filtering to ensure data quality:

  * **Minimum Length:** 60 steps (Episodes shorter than this are skipped).
  * **Maximum Count:** 10,000 episodes.

### Prompt Strategy

The VLM is prompted to generate **concise, object-centric** descriptions.

  * *Prompt Constraints:* "Focus on concrete physical objects... Do NOT describe the whole room... Do NOT mention the robot itself... Write one or two concise sentences.".

## Output Structure

Results are saved to `/vision/u/yinhang/pre_process/task2_outputs`.

For each processed episode (e.g., `episode_000000`), two files are generated:

1.  **Image:** `episode_000000.png` (The first frame used for captioning).
2.  **Metadata:** `episode_000000.json` containing:

<!-- end list -->

```json
{
  "episode_id": "episode_000000",
  "instruction": "put the block in the bowl",
  "scene_caption": "The green block is on the white table next to a blue ceramic bowl.",
  "combined_description": "The green block is on the white table next to a blue ceramic bowl. The robot is instructed to: put the block in the bowl",
  "image_path": ".../episode_000000.png",
  "num_steps": 120
}
```

```

