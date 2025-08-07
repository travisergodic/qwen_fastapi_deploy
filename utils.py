import io
import base64
from PIL import Image


def encode_image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"


def decode_base64_image(image_str: str) -> Image.Image:
    try:
        if image_str.startswith("data:image"):
            image_str = image_str.split(",")[1]
        return Image.open(io.BytesIO(base64.b64decode(image_str)))
    except:
        raise ValueError("fail to decode image")


def mask_base64_images(messages: list) -> list:
    """将消息中的 base64 图片替换为占位符 <image>"""
    masked = []
    for msg in messages:
        new_msg = msg.copy()
        if isinstance(new_msg.get("content"), list):
            new_msg["content"] = []
            for item in msg["content"]:
                if item.get("type") == "image":
                    new_msg["content"].append({"type": "image", "image": "<image>"})
                else:
                    new_msg["content"].append(item)
        masked.append(new_msg)
    return masked