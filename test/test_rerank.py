import requests

# 设置 reranker 接口地址

# 构造 payload
payload = {
    "instruction": "Given a web search query, retrieve relevant passages that answer the query",
    "query": "What is the capital of China?",
    "docs": [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ],
    "prefix": "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n",
    "suffix": "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "max_length": 2048
}

# 发送 POST 请求

for url in ("http://127.0.0.1:8500/qwen3-rerank-0.6b", "http://127.0.0.1:8500/qwen3-rerank-4b"):
    response = requests.post(url, json=payload)

# 打印响应结果
print("Status:", response.status_code)
print("Response:", response.json())