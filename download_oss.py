import os
import oss2

# ✅ 填入你的配置
access_key_id = ''
access_key_secret = ''
bucket_name = 'travis-hf-model'
endpoint = 'oss-cn-hangzhou.aliyuncs.com' 
oss_key = 'qwen25-vl/Qwen2.5-VL-7B-Instruct.tar'  # OSS 上的路径
local_dir = ''

# 初始化认证与 bucket
auth = oss2.Auth(access_key_id, access_key_secret)
bucket = oss2.Bucket(auth, endpoint, bucket_name)

# 下载
for oss_key in (
    'qwen25-vl/Qwen2.5-VL-7B-Instruct.tar', 
    'qwen3/models--Qwen--Qwen3-Embedding-0.6B.tar',
    'qwen3/models--Qwen--Qwen3-Reranker-0.6B.tar',
    'qwen3/models--Qwen--Qwen3-Reranker-4B.tar',
    'jina/models--jinaai--jina-embeddings-v3.tar'
):
    local_file = os.path.join(local_dir, os.path.basename(oss_key))
    bucket.get_object_to_file(oss_key, local_file)

    print(f"✅ 下载成功: {oss_key} -> {local_file}")