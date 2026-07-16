import os

# 即梦AI官方固定常量
HOST = "visual.volcengineapi.com"
ACTION = "CVProcess"
VERSION = "2022-08-01"
SERVICE = "cv"
REGION = "cn-north-1"
FIX_REQ_KEY = "jimeng_high_aes_general_v21_L"

class GlobalConfig:
    def __init__(self):
        self.raw_ak = os.getenv("JIMENG_AK", "").strip()
        self.raw_sk = os.getenv("JIMENG_SK", "").strip()
        # 自动清除AK首尾斜杠，避免双斜杠报错
        self.ak = self.raw_ak.strip("/")
        self.sk = self.raw_sk

    # 校验密钥是否存在
    def is_valid(self):
        return len(self.ak) > 0 and len(self.sk) > 0

# 全局单例配置，全项目共用
CONF = GlobalConfig()