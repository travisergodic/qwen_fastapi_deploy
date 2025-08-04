import logging

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


logger = logging.getLogger(__name__)


class Qwen25_VL_Predictor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name_or_path = cfg.model_name_or_path
        self.attn_implementation = cfg.attn_implementation

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16,
            attn_implementation=self.attn_implementation,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)

        logger.info(f"Load {self.model_name_or_path} sucessfully")
     
    def predict(self, request):
        messages_dict = [m.dict() for m in request.messages]

        text = self.processor.apply_chat_template(
            messages_dict, 
            tokenize=False, 
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages_dict)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # 模型推理
        generated_ids = self.model.generate(
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
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        return output_text