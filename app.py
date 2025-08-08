import json
import time
import uuid

from omegaconf import OmegaConf
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from predictor import PREDICTOR
from utils import mask_base64_images
from logger_helper import setup_logger
from schemas import ChatRequest, EmbeddingRequest, JinaV3EmbeddingRequest, RerankRequest

logger = setup_logger()
app = FastAPI()


cfg = OmegaConf.load("configs/qwen.yaml")
open(cfg["record_path"], "w").close()

model_cfg = cfg["models"]
record_path = cfg["record_path"]

MODEL = {model_name:PREDICTOR.build(**model_cfg[model_name]) for model_name in model_cfg}

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

        messages_dict = [m.dict() for m in request.messages]
        record.update({
            "status": "error",
            "input": mask_base64_images(messages_dict),
            "error": error_msg,
            "duration": round(duration, 2)
        })
        return JSONResponse(content={"id": request_id, "error": error_msg}, status_code=500)

    finally:
        with open(record_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")



@app.post("/qwen3-embed-0.6b")
async def embed(request: EmbeddingRequest):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    record = {"id": request_id}

    try:
        output = MODEL["qwen3-embed-0.6b"].predict(request=request)
        output = output.tolist()  # tensor 转 list，确保 JSON 可序列化
        duration = time.time() - start_time
        logger.info(f"[{request_id}] SUCCESS in {duration:.2f}s")

        record.update({
            "status": "success",
            "input": request.dict(),
            "duration": round(duration, 2),
            # "output": output
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
        with open(record_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.post("/qwen3-rerank-0.6b")
async def rerank_06b(request: RerankRequest):
    return await handle_rerank(request, model_key="qwen3-rerank-0.6b")


@app.post("/qwen3-rerank-4b")
async def rerank_4b(request: RerankRequest):
    return await handle_rerank(request, model_key="qwen3-rerank-4b")


@app.post("/jinav3-embed")
async def embed(request: JinaV3EmbeddingRequest):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    record = {"id": request_id}

    try:
        # 調用模型
        output = MODEL["jinav3-embed"].predict(request=request)
        output = output.tolist()  # tensor -> list 確保 JSON 可序列化
        duration = time.time() - start_time
        logger.info(f"[{request_id}] SUCCESS in {duration:.2f}s")

        record.update({
            "status": "success",
            "input": request.dict(),
            "duration": round(duration, 2),
            # "output": output 
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
        # 寫紀錄檔
        with open(record_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def handle_rerank(request: RerankRequest, model_key: str):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    record = {"id": request_id}

    try:
        scores = MODEL[model_key].predict(request=request)
        duration = time.time() - start_time
        logger.info(f"[{request_id}] SUCCESS in {duration:.2f}s")

        record.update({
            "status": "success",
            "input": request.dict(),
            "output": scores,
            "duration": round(duration, 2)
        })
        return JSONResponse(content={"id": request_id, "output": scores})

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
        with open(record_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")