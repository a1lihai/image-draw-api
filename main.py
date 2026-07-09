import hmac
import hashlib
import json
import requests
import logging
from datetime import datetime
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 全局固定常量（模库统一硬编码，不拆分）
HOST = "visual.volcengineapi.com"
ACTION = "CVProcess"
VERSION = "2022-08-01"
SERVICE = "cv"
REGION = "cn-north-1"
REQ_KEY = "jimeng_high_aes_general_v21_L"

# 环境读取
AK_RAW = os.getenv("JIMENG_AK", "").strip()
SK_RAW = os.getenv("JIMENG_SK", "").strip()
AK = AK_RAW.strip("/")

class DrawReq(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 832
    height: int = 1216
    num: int = 1

# 标准V4签名（模库同款分层推导）
def sign_v4(sk: str, dt_short: str, xdt: str, canonical_req: str):
    kd = hmac.new(sk.encode(), dt_short.encode(), hashlib.sha256).digest()
    kr = hmac.new(kd, REGION.encode(), hashlib.sha256).digest()
    ks = hmac.new(kr, SERVICE.encode(), hashlib.sha256).digest()
    ksig = hmac.new(ks, b"request", hashlib.sha256).digest()
    cr_hash = hashlib.sha256(canonical_req.encode()).hexdigest()
    sign_str = f"HMAC-SHA256\n{xdt}\n{dt_short}/{REGION}/{SERVICE}/request\n{cr_hash}"
    return hmac.new(ksig, sign_str.encode(), hashlib.sha256).hexdigest()

@app.post("/api/v1/image/generate")
def gen_img(body: DrawReq):
    if not AK or not SK_RAW:
        return {"code":-1,"detail":"缺少AK/SK环境变量"}
    
    # 双重兜底时间，模库标准校验
    now = datetime.utcnow()
    x_date = now.strftime("%Y%m%dT%H:%M:%SZ")
    date_short = now.strftime("%Y%m%d")
    # 双重校验：空值 + 长度不足8位直接替换固定日期
    if not date_short or len(date_short) != 8:
        date_short = "20260708"

    # 请求体
    payload = {
        "ReqKey": REQ_KEY,
        "StableDiffusion":{
            "Prompt":body.prompt,
            "NegativePrompt":body.negative_prompt,
            "ImageSize":f"{body.width}*{body.height}",
            "Num":body.num
        }
    }
    body_bin = json.dumps(payload, separators=(",",":"), ensure_ascii=False).encode("utf-8")
    body_sha = hashlib.sha256(body_bin).hexdigest()

    query = f"Action={ACTION}&Version={VERSION}"
    canonical_header = (
        f"content-type:application/json; charset=utf-8\n"
        f"host:{HOST}\n"
        f"x-content-sha256:{body_sha}\n"
        f"x-date:{x_date}\n"
    )
    signed_keys = "content-type;host;x-content-sha256;x-date"
    canonical_req = "\n".join([
        "POST","/",query,canonical_header.rstrip("\n"),signed_keys,body_sha
    ])

    sig = sign_v4(SK_RAW, date_short, x_date, canonical_req)
    # 核心修复：scope一次性完整拼接，日期强制嵌入，永远不会丢失
    auth = (
        f"HMAC-SHA256 "
        f"Credential={AK}/{date_short}/{REGION}/{SERVICE}/request,"
        f"SignedHeaders={signed_keys},"
        f"Signature={sig}"
    )

    headers = {
        "Content-Type":"application/json; charset=utf-8",
        "X-Content-Sha256":body_sha,
        "X-Date":x_date,
        "Authorization":auth
    }
    url = f"https://{HOST}?{query}"
    resp = requests.post(url, headers=headers, data=body_bin, timeout=120)
    res_json = resp.json()
    logger.info(res_json)

    err = res_json.get("ResponseMetadata",{}).get("Error")
    if err:
        return {"code":-1,"detail":f"{err['Code']}:{err['Message']}"}
    
    img_urls = [i["ImageUrl"] for i in res_json["Result"]["StableDiffusion"]["Images"]]
    return {"code":0,"msg":"success","data":{"image_urls":img_urls}}

# 前端页面
@app.get("/", response_class=HTMLResponse)
def index():
    html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>AI生图 | 即梦AI</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;font-family:system-ui}
body{max-width:600px;margin:30px auto;padding:0 15px}
.box{border:1px solid #eee;border-radius:12px;padding:20px;margin-bottom:16px}
h2{margin-bottom:16px}
textarea,input{width:100%;padding:10px;border:1px solid #ddd;border-radius:8px;margin-bottom:12px}
textarea{height:100px;resize:none}
button{width:100%;padding:12px;background:#2563eb;color:#fff;border:none;border-radius:8px;font-size:16px}
#result{margin-top:20px;white-space:pre-wrap;color:#c00}
</style>
</head>
<body>
<div class="box">
<h2>AI生图 | 即梦AI</h2>
<textarea id="prompt" placeholder="正向提示词"></textarea>
<textarea id="neg" placeholder="反向提示词"></textarea>
<div style="display:flex;gap:10px">
<input type="number" id="w" value="832">
<input type="number" id="h" value="1216">
</div>
<button onclick="runDraw()">生成图片</button>
<div id="result"></div>
</div>
<script>
async function runDraw(){
    const box = document.getElementById("result");
    box.innerText = "请求中...";
    const req = {
        prompt:document.getElementById("prompt").value,
        negative_prompt:document.getElementById("neg").value,
        width:+document.getElementById("w").value,
        height:+document.getElementById("h").value,
        num:1
    };
    const r = await fetch("/api/v1/image/generate",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify(req)
    });
    const d = await r.json();
    box.innerText = JSON.stringify(d,null,2);
}
</script>
</body>
</html>
"""
    return html

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT",8000)))