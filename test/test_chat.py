import os
import sys
import requests
sys.path.insert(0, os.getcwd())

from utils import encode_image_to_base64


url = "http://127.0.0.1:8500/qwen2.5-vl-7b"

payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": encode_image_to_base64("assets/image.jpg")
                },
                {
                    "type": "text",
                    "text": "描述图片。"
                }
            ]
        }
    ],
    "max_new_tokens": 1000,
    "temperature": 1.0,
    "top_p": 1.0,
    "do_sample": False
}

# 用 json 傳遞正確的 Content-Type
response = requests.post(url, json=payload)

print("Status:", response.status_code)
print("Response:", response.json())