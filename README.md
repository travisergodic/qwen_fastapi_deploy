# Qwen2.5-VL 部署

## 安装
1. **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```
2. **安装 Flash Attention**: 根据 python, torch, cuda 版本，安装特定的 **flash attention** 版本（参考[链接](https://github.com/Dao-AILab/flash-attention/releases/?fbclid=IwY2xjawL4_0lleHRuA2FlbQIxMABicmlkETFyTzNNTXFxTW5RZFZOMXpPAR4UmnmOZ6yEETh7cB2Bd2GvzU-kw2jw48YiKGEoNK9cudL6Vc8CapqG2dJYfA_aem_o9J-KtD4Zvbo23v0cb4Q6Q)），范例如下：
    ```bash
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
    ```

## 执行
1. **启动服务**
    ```bash
    uvicorn qwen25_vl_app:app --host 0.0.0.0 --port 8000
    ```

2. **访问服务**
    ```python
    import requests
    import json

    url = "http://localhost:8000/qwen25_vl"

    # 文本消息结构（注意 image 字段可以是占位）
    message_dict = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "upload.jpg"},  # 占位，服务端会替换
                {"type": "text", "text": "请确认图中每一列都打了一个勾。"}
            ]
        }
    ]

    files = {
        "image": open("local_image.jpg", "rb"),
    }
    data = {
        "message_json": json.dumps(message_dict)
    }

    response = requests.post(url, data=data, files=files)

    print("Status:", response.status_code)
    print("Response:", response.json())
    ```   