import json
import time
import uuid

from omegaconf import OmegaConf
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from schemas import ChatRequest, EmbeddingRequest, RerankRequest
from predictor import PREDICTOR
from utils import mask_base64_images
from logger_helper import setup_logger


logger = setup_logger()
app = FastAPI()


cfg = OmegaConf.load("configs/qwen.yaml")

open(cfg["record_path"], "w").close()


MODEL = [PREDICTOR.build(type=model_name, **cfg[model_name]) for model_name in cfg]

@app.post("/qwen2.5-vl-7b")
async def chat(request: ChatRequest):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    record = {"id": request_id}
    try:
        output_text = MODEL["qwen2.5-vl-7b"].predict(request=request)
        duration = time.time() - start_time 
        logger.info(f"[{request_id}] SUCCESS in {duration:.2f}s")

        messages_dict = [m.dict() for m in request.messages]
        record.update({
            "status": "success",
            "input": mask_base64_images(messages_dict),
            "output": output_text,
            "duration": round(duration, 2)
        })
        return JSONResponse(content={"id": request_id, "output": output_text})

    except Exception as e:
        duration = time.time() - start_time 
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
        with open(cfg["record_path"], "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")



@app.post("/qwen3-embed")
async def embed(request: EmbeddingRequest):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    record = {"id": request_id}

    try:
        output = MODEL["qwen3-embed"].predict(request=request)
        output = output.tolist()  # tensor 转 list，确保 JSON 可序列化
        duration = time.time() - start_time
        logger.info(f"[{request_id}] SUCCESS in {duration:.2f}s")

        record.update({
            "status": "success",
            "input": request.dict(),
            "output": output,
            "duration": round(duration, 2)
        })
        return JSONResponse(content={"id": request_id, "output": output})

    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        logger.error(f"[{request_id}] ERROR after {duration:.2f}s: {error_msg}")

        record.update({
            "status": "error",
            "input": request.dict(),
            "error": error_msg,
            "duration": round(duration, 2)
        })
        return JSONResponse(content={"id": request_id, "error": error_msg}, status_code=500)

    finally:
        with open(cfg["record_path"], "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")



@app.post("/qwen3-rerank")
async def rerank(request: RerankRequest):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    record = {"id": request_id}

    try:
        scores = MODEL["qwen3-rerank"].predict(request=request)
        duration = time.time() - start_time
        logger.info(f"[{request_id}] SUCCESS in {duration:.2f}s")

        record.update({
            "status": "success",
            "input": request.dict(),
            "output": scores,
            "duration": round(duration, 2)
        })
        return JSONResponse(content={"id": request_id, "scores": scores})

    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        logger.error(f"[{request_id}] ERROR after {duration:.2f}s: {error_msg}")

        record.update({
            "status": "error",
            "input": request.dict(),
            "error": error_msg,
            "duration": round(duration, 2)
        })
        return JSONResponse(content={"id": request_id, "error": error_msg}, status_code=500)

    finally:
        with open(cfg["record_path"], "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")