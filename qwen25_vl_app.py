import json
import time
import uuid

from omegaconf import OmegaConf
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from schemas import ChatRequest
from predictor import Qwen25_VL_Predictor
from utils import mask_base64_images
from logger_helper import setup_logger


logger = setup_logger()
app = FastAPI()


cfg = OmegaConf.load("config.yaml")

open(cfg.record_path, "w").close()

qwen25_vl_predictor = Qwen25_VL_Predictor(cfg)


@app.post("/qwen25-vl-7b")
async def chat(request: ChatRequest):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    record = {"id": request_id}
    try:
        output_text = qwen25_vl_predictor.predict(request=request)
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
        with open(cfg.record_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")