import io
import base64
from PIL import Image

def decode_base64_image(image_str: str) -> Image.Image:
    if image_str.startswith("data:image"):
        image_str = image_str.split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(image_str)))