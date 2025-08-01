import yaml

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from utils import decode_base64_image
from schemas import ChatRequest


with open("configs/qwen25vl.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


app = FastAPI()

# Load model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    CONFIG["MODEL_NAME_OR_PATH"],
    torch_dtype=torch.float16,
    attn_implementation=CONFIG["ATTN_IMPLEMENTATION"],
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(CONFIG["MODEL_NAME_OR_PATH"])


@app.post("/qwen25_vl")
async def chat(request: ChatRequest):
    try:
        # for message in request.messages:
        #     for item in message.content:
        #         if item.type == "image" and isinstance(item.image, str):
        #             print(item.image[:100])
        #             item.image = decode_base64_image(item.image)
        #             print(type(item.image))

        messages_dict = [m.dict() for m in request.messages]

        print(messages_dict)

        # 準備文字與視覺輸入
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
        generated_ids = model.generate(**inputs, max_new_tokens=request.max_new_tokens, use_cache=True)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return JSONResponse(content={"output": output_text[0]})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)