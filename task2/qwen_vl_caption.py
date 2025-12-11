import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # 防止 transformers 去 import TF

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

print(f"Loading Qwen-VL model: {MODEL_NAME} ...")

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,          # 等价于原来的 torch_dtype
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

@torch.no_grad()
def qwen_vl_caption(img_np: np.ndarray) -> str:
    """
    img_np: numpy array (H, W, 3), RGB, uint8
    return: caption string (one concise paragraph)
    """
    if img_np.dtype != np.uint8:
        img_np = img_np.astype(np.uint8)

    image = Image.fromarray(img_np)

    # prompt = (
        
    #     "You see a robot manipulation scene with a tabletop and a robot arm. "
    #     "Describe in one concise paragraph: "
    #     "the tabletop objects (names, colors, materials, approximate positions), "
    #     "any containers (bowls, cups, plates, trays), and any visible furniture "
    #     "(tables, drawers, shelves). Do not use bullet points. For example, "
    #     "The mug is white outside and orange inside and is on the white desk. There is a laptop with a mouse on top on the right of the table and an earbud case."
    #     "Do not mention that you are an assistant or describe the instructions."
    # )

    
    # prompt = (
    #     "You see a robot manipulation scene with a tabletop and a robot arm. "
    #     "Describe in one concise paragraph only what is clearly visible inside the image frame. "
    #     "Focus on the tabletop objects (names, colors, materials, approximate positions) "
    #     "and containers (bowls, cups, plates, trays). "
    #     "If furniture like shelves or drawers are not clearly visible, do not mention them. "
    #     "Do not guess or imagine objects that are not clearly visible. "
    #     "For example, The mug is white outside and orange inside and is on the white desk. "
    #     "There is a laptop with a mouse on top on the right of the table and an earbud case. "
    #     "Do not mention that you are an assistant or describe the instructions."
    # )

    # prompt = (
    #     "You are describing a robot manipulation scene. "
    #     "Focus only on objects that are clearly visible in the robot's workspace "
    #     "(the area where the robot could interact with items). "
    #     "Do NOT describe the room, walls, monitors, chairs, shelves, large furniture, or anything far away. "
    #     "Do NOT guess or hallucinate objects that are not clearly shown. "
    #     "Write one concise, factual sentence describing the visible objects, their colors, and their approximate relative positions.\n\n"
    #     "Example of a good description:\n"
    #     "“The mug is white outside and orange inside and sits near the left edge of the white desk, "
    #     "with a laptop and mouse placed on the right side of the workspace.”\n\n"
    #     "Now describe the objects visible in this scene:"
    # )

    # prompt = (
    #     "You are helping to annotate a robot manipulation dataset. "
    #     "Look carefully at the image and describe ONLY the physical objects that are clearly visible "
    #     "near the robot gripper or on the main supporting surface (such as a table, desk, counter, or shelf). "
    #     "Focus on concrete objects, their colors, and approximate positions "
    #     "(for example: 'on the left side of the desk', 'near the robot gripper', 'in the white bowl'). "
    #     "Do NOT talk about the room, office, workspace, environment, monitors, chairs, walls, lights, "
    #     "or anything far in the background. "
    #     "Do NOT mention the robot itself, cameras, or any tools that are mostly outside the frame. "
    #     "If you are unsure an object is present, do NOT mention it. No guessing. "
    #     "Write exactly ONE concise sentence. "
    #     "Example of a good answer: "
    #     "'The mug is white outside and orange inside and sits on the white desk next to a closed laptop, a black mouse, and a small white earbud case.' "
    #     "Answer with only that one sentence describing the visible objects; do NOT add any role words like 'assistant'."
    # )


    
    # prompt = (
    #     "You are describing a robot manipulation scene."
    #     "Describe in one concise paragraph only what is clearly visible inside the image frame. "
    #     "Focus on concrete objects, their colors, and approximate positions "
    #     "Do NOT talk about the room, office, workspace, environment, monitors, chairs, walls, lights, "
    #     "or anything far in the background. "
    #     "Do NOT mention the robot itself, cameras, or any tools that are mostly outside the frame. "
    #     "If you are unsure an object is present, do NOT mention it. No guessing. "        
    #     "If furniture like shelves or drawers are not clearly visible, do not mention them. "
    #     "Do not guess or imagine objects that are not clearly visible. "
    #     "For example, The mug is white outside and orange inside and is on the white desk. "
    #     "There is a laptop with a mouse on top on the right of the table and an earbud case. "
    #     "Do not mention that you are an assistant or describe the instructions."
    # )

    #so far the best one:
    # prompt = (
    #     "You are describing a robot manipulation scene. "
    #     "Describe ONLY what is clearly visible inside the image frame. "
    #     "Focus on concrete physical objects, their colors, and approximate positions. "
    #     "Do NOT talk about the room, office, workspace, environment, monitors, chairs, walls, lights, "
    #     "or anything far in the background. "
    #     "Do NOT mention the robot itself, cameras, or any tools that are mostly outside the frame. "
    #     "If you are unsure an object is present, do NOT mention it. No guessing. "
    #     "Write exactly ONE concise sentence, no more than 50 English words. "
    #     "Example of a good answer: "
    #     "\"The mug is white outside and orange inside and sits on the white desk next to a closed laptop, a black mouse, and a small white earbud case.\" "
    #     "Answer with only that one sentence describing the visible objects."
    # )

    prompt = (
        "You are describing a robot manipulation scene. "
        "Describe only what is clearly visible inside the image frame. "
        "Focus on concrete physical objects, their colors, shapes, and approximate positions. "
        "You may briefly mention the main supporting surface such as a table, desk, counter, or tray, "
        "but do NOT describe the whole room, office, walls, ceiling, lights, or far background. "
        "Do NOT mention the robot itself, cameras, or tools that are mostly outside the frame. "
        "If you are unsure an object is present, do NOT mention it. No guessing. "
        "Write one or two concise sentences, together no more than 45 English words. "
        "Example of a good answer: "
        "\"The mug is white outside and orange inside and sits on the white desk next to a closed laptop, "
        "a black mouse, and a small white earbud case.\" "
        "Answer with only those sentences describing the visible objects."
    )


    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    chat_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        text=[chat_text],
        images=[image],
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=160,
        do_sample=False,
    )
    
    caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    print("[RAW CAPTION]", repr(caption[:200]))

    # ---- Step 1: 以 prompt 作为 anchor ----
    if prompt in caption:
        # 得到 prompt 后面的所有内容
        after_prompt = caption.split(prompt, 1)[-1].lstrip()
    else:
        after_prompt = caption  # 极少情况 prompt 不在 decode 里

    # ---- Step 2: 寻找 assistant 的分割点 ----
    # Qwen 输出通常是 "assistant" 或 "assistant\n"
    for sep in ["assistant\n", "assistant:", "assistant :"]:
        if sep in after_prompt:
            after_prompt = after_prompt.split(sep, 1)[-1].lstrip()
            break

    final_caption = after_prompt.strip()

    print("[AFTER CLEAN]", repr(final_caption[:200]))
    return final_caption
