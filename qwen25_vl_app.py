import json
import time
import yaml
import uuid
from datetime import datetime

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from utils import decode_base64_image, mask_base64_images
from schemas import ChatRequest
from github.logger_helper import setup_logger

logger = setup_logger()


with open("configs/qwen25vl.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

open(CONFIG["RECORD_PATH"], "w").close()

app = FastAPI()

# Load model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    CONFIG["MODEL_NAME_OR_PATH"],
    torch_dtype=torch.float16,
    attn_implementation=CONFIG["ATTN_IMPLEMENTATION"],
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(CONFIG["MODEL_NAME_OR_PATH"])

logger.info(f"Load {CONFIG['MODEL_NAME_OR_PATH']} sucessfully")


@app.post("/qwen25-vl-7b")
async def chat(request: ChatRequest):
    request_id = str(uuid.uuid4())
    start_time = time.time()  # ✅ 开始计时
    record = {"id": request_id}
    try:
        messages_dict = [m.dict() for m in request.messages]

        # 准备输入
        text = processor.apply_chat_template(messages_dict, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages_dict)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # 模型推理
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=request.max_new_tokens, 
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample,
            use_cache=request.use_cache
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        duration = time.time() - start_time  # ✅ 结束计时
        logger.info(f"[{request_id}] SUCCESS in {duration:.2f}s")

        record.update({
            "status": "success",
            "input": mask_base64_images(messages_dict),
            "output": output_text,
            "duration": round(duration, 2)
        })

        return JSONResponse(content={"id": request_id, "output": output_text})

    except Exception as e:
        duration = time.time() - start_time  # ✅ 错误也记录时间
        error_msg = str(e)
        logger.error(f"[{request_id}] ERROR after {duration:.2f}s: {error_msg}")

        record.update({
            "status": "error",
            "input": mask_base64_images(messages_dict),
            "error": error_msg,
            "duration": round(duration, 2)
        })

        return JSONResponse(content={"id": request_id, "error": error_msg}, status_code=500)

    finally:
        with open(CONFIG["RECORD_PATH"], "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")