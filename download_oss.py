import oss2

# ✅ 填入你的配置
access_key_id = ''
access_key_secret = ''
bucket_name = 'travis-hf-model'
endpoint = 'oss-cn-hangzhou.aliyuncs.com' 
oss_key = 'qwen25-vl/Qwen2.5-VL-7B-Instruct.tar'  # OSS 上的路径
local_file = '/home/bglab/htw/Qwen2.5-VL-7B-Instruct.tar'

# 初始化认证与 bucket
auth = oss2.Auth(access_key_id, access_key_secret)
bucket = oss2.Bucket(auth, endpoint, bucket_name)

# 下载
bucket.get_object_to_file(oss_key, local_file)

print(f"✅ 下载成功: {oss_key} -> {local_file}")