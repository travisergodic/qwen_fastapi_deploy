import requests

url = "http://127.0.0.1:8500/jinav3-embed"

# 构造 payload
payload = {
    "texts": [
        "Follow the white rabbit.",  # English
        "Sigue al conejo blanco.",  # Spanish
        "Suis le lapin blanc.",  # French
        "跟着白兔走。",  # Chinese
        "اتبع الأرنب الأبيض.",  # Arabic
        "Folge dem weißen Kaninchen.",  # German
    ],
    "task_label": "text-matching"
}

# 发送 POST 请求
response = requests.post(url, json=payload)

# 打印响应结果
print("Status:", response.status_code)
print("Response:", response.json())