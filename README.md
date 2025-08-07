# Qwen 部署

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
    uvicorn app:app --host 0.0.0.0 --port 8500 --reload
    ```

2. **访问服务**
    ```python
    import requests
    from utils import encode_image_to_base64

    url = "http://192.168.137.26:8500/qwen2.5-vl"

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": encode_image_to_base64("/content/demo2.jpg")
                    },
                    {
                        "type": "text",
                        "text": "请确认图中每一列都打了一个勾。"
                    }
                ]
            }
        ],
        "max_new_tokens": 1000,
        "temperature": 1.0,
        "top_p": 1.0,
        "do_sample": True
    }

    # 用 json 傳遞正確的 Content-Type
    response = requests.post(url, json=payload)

    print("Status:", response.status_code)
    print("Response:", response.json())
    ```   