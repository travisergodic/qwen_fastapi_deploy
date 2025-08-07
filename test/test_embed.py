import requests

# 设置 embedding API 地址
url = "http://127.0.0.1:8500/qwen3-embed"  # 确保端口与服务一致

# 构造请求数据
payload = {
    "texts": ["The capital of China is Beijing."],
    "prompt_template": "{query}",
    "max_length": 2048
}

# 发送 POST 请求
response = requests.post(url, json=payload)

# 打印响应信息
print("Status:", response.status_code)
print("Response:", response.json())
