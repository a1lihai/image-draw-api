import os
import hmac
import hashlib
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

AK = os.getenv("JIMENG_AK")
SK = os.getenv("JIMENG_SK")
REGION = os.getenv("JIMENG_REGION", "cn-beijing")
REQ_KEY = os.getenv("JIMENG_REQ_KEY", "jimeng_high_aes_general_v21_L")

if not AK or not SK:
    print("错误：JIMENG_AK 或 JIMENG_SK 没有从 .env 读取到")
    exit(1)

service = "visual"
version = "2018-08-01"
action = "GenerateImage"
date = datetime.utcnow().strftime("%Y%m%dT%H:%M:%SZ")

payload = {
    "ReqKey": REQ_KEY,
    "Prompt": "一只可爱的小猫",
    "NegativePrompt": "模糊,低画质,畸形,水印,文字",
    "ImageSize": "768x768",
    "Num": 1
}

canonical_str = f"{service}\n{version}\n{action}\n{date}"
hmac_obj = hmac.new(SK.encode("utf-8"), canonical_str.encode("utf-8"), hashlib.sha256)
signature = hmac_obj.hexdigest()

headers = {
    "X-Date": date,
    "Authorization": f'HMAC-SHA256 Credential={AK},SignedHeaders=X-Date,Signature={signature}',
    "Content-Type": "application/json; charset=utf-8"
}

url = f"https://visual.volcengineapi.com?Action={action}&Version={version}"

print("正在测试即梦接口...")
print(f"AK: {AK}")
print(f"SK前10位: {SK[:10]}...")
print(f"REQ_KEY: {REQ_KEY}")

resp = requests.post(url, headers=headers, json=payload, timeout=120)

print(f"\n状态码: {resp.status_code}")
print(f"返回内容: {resp.text}")