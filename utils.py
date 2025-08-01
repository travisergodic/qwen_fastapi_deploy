import io
import base64
from PIL import Image


def encode_image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"


def decode_base64_image(image_str: str) -> Image.Image:
    if image_str.startswith("data:image"):
        image_str = image_str.split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(image_str)))