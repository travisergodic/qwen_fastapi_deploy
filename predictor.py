import logging

import torch
from torch import Tensor
from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    AutoModel,
    AutoProcessor, 
    AutoTokenizer, 
    AutoModelForCausalLM
)
from qwen_vl_utils import process_vision_info

from registry import PREDICTOR


logger = logging.getLogger(__name__)


def format_reranker_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
    return output


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    

@PREDICTOR.register("qwen2.5-vl")
class Qwen25_VL_Predictor:
    def __init__(self, model_name_or_path, attn_implementation, device_map):
        self.model_name_or_path = model_name_or_path
        self.attn_implementation = attn_implementation
        self.device_map = device_map

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16,
            attn_implementation=self.attn_implementation,
            device_map=self.device_map
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


@PREDICTOR.register("qwen3-rerank")
class Qwen3RerankerPredictor:
    def __init__(self, model_name_or_path, attn_implementation, device_map):
        self.model_name_or_path = model_name_or_path
        self.attn_implementation = attn_implementation
        self.device_map = device_map
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            attn_implementation=self.attn_implementation,
            torch_dtype=torch.float16,
            device_map=self.device_map
        ).eval()
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        logger.info(f"Load {self.model_name_or_path} sucessfully")

    @torch.no_grad()
    def predict(self, request):
        instruction = request.instruction
        query = request.query
        docs = request.docs
        max_length = request.max_length
        prefix = request.prefix
        suffix = request.suffix

        pairs = [format_reranker_instruction(instruction, query, doc) for doc in docs]
        inputs = self.process_inputs(pairs, prefix, suffix, max_length)
        scores = self.compute_logits(inputs)
        return scores


    def compute_logits(self, inputs, **kwargs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def process_inputs(self, pairs, prefix, suffix, max_length):
        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs
    

@PREDICTOR.register("qwen3-embed")
class Qwen3EmbeddingPredictor:
    def __init__(self, model_name_or_path, attn_implementation, device_map):
        self.model_name_or_path = model_name_or_path
        self.attn_implementation = attn_implementation
        self.device_map = device_map
        self.model = AutoModel.from_pretrained(
            self.model_name_or_path,
            attn_implementation=self.attn_implementation,
            torch_dtype=torch.float16,
            device_map=self.device_map
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, padding_side='left')
        logger.info(f"Load {self.model_name_or_path} sucessfully")

    @torch.no_grad()
    def predict(self, request):
        texts = request.texts
        prompt_template = request.prompt_template
        max_length = request.max_length

        input_texts = [prompt_template.format(query=text) for text in texts]
        batch_dict = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_dict.to(self.model.device)
        outputs = self.model(**batch_dict)
        return last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    

@PREDICTOR.register("jinav3-embed")
class JinaEmbeddingV3Predictor:
    def __init__(self, model_name_or_path, device_map):
        self.model_name_or_path = model_name_or_path
        self.device_map = device_map
        self.model = AutoModel.from_pretrained(
            self.model_name_or_path, 
            device_map=self.device_map,
            torch_dtype=torch.float16
        ).eval()

    @torch.no_grad()
    def predict(self, request):
        texts = request.texts
        task = request.task
        return self.model.encode(texts, task=task)
