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

# -------------------------- 全局日志 --------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------------- FastAPI实例 --------------------------
app = FastAPI(title="Jimeng CVProcess Official Demo")

# -------------------------- 官方固定常量（不可修改） --------------------------
HOST = "visual.volcengineapi.com"
ACTION = "CVProcess"
VERSION = "2022-08-01"
SERVICE = "cv"
REGION = "cn-north-1"
# 官方固定ReqKey，无需控制台创建应用
REQ_KEY = "jimeng_high_aes_general_v21_L"

# -------------------------- 环境配置读取 --------------------------
class GlobalConfig:
    def __init__(self):
        self.ak = os.getenv("JIMENG_AK", "").strip()
        self.sk = os.getenv("JIMENG_SK", "").strip()

CONF = GlobalConfig()

# -------------------------- 前端入参模型 --------------------------
class ImageDrawReq(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 832
    height: int = 1216
    num: int = 1

# -------------------------- 火山官方原版V4签名函数（照搬官方仓库demo） --------------------------
def get_signing_key(secret_key: str, date_stamp: str, region: str, service: str):
    k_date = hmac.new(secret_key.encode("utf-8"), date_stamp.encode("utf-8"), hashlib.sha256).digest()
    k_region = hmac.new(k_date, region.encode("utf-8"), hashlib.sha256).digest()
    k_service = hmac.new(k_region, service.encode("utf-8"), hashlib.sha256).digest()
    k_signing = hmac.new(k_service, b"request", hashlib.sha256).digest()
    return k_signing

def calc_authorization(ak: str, sk: str, x_date: str, date_stamp: str, canonical_req: str):
    # 1. 计算规范请求哈希
    cr_hash = hashlib.sha256(canonical_req.encode("utf-8")).hexdigest()
    # 2. 构造待签名字符串（官方固定格式）
    credential_scope = f"{date_stamp}/{REGION}/{SERVICE}/request"
    string_to_sign = f"HMAC-SHA256\n{x_date}\n{credential_scope}\n{cr_hash}"
    # 3. 分层推导签名密钥
    signing_key = get_signing_key(sk, date_stamp, REGION, SERVICE)
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    # 4. 拼接单行鉴权头，逗号无空格、无任何换行（核心修复报错）
    auth_header = (
        f"HMAC-SHA256 "
        f"Credential={ak.strip('/')}/{credential_scope},"
        f"SignedHeaders=content-type;host;x-content-sha256;x-date,"
        f"Signature={signature}"
    )
    return auth_header

# -------------------------- 官方标准即梦绘图调用逻辑 --------------------------
def jimeng_cvprocess_draw(req: ImageDrawReq) -> List[str]:
    ak = CONF.ak
    sk = CONF.sk
    if not ak or not sk:
        raise Exception("缺失环境变量 JIMENG_AK / JIMENG_SK")

    # UTC标准时间，兜底防止空日期
    now_utc = datetime.utcnow()
    x_date = now_utc.strftime("%Y%m%dT%H:%M:%SZ")
    date_stamp = now_utc.strftime("%Y%m%d")
    if not date_stamp:
        date_stamp = "20260708"

    # 1. 构造业务Body（CVProcess标准结构）
    body = {
        "ReqKey": REQ_KEY,
        "StableDiffusion": {
            "Prompt": req.prompt,
            "NegativePrompt": req.negative_prompt,
            "ImageSize": f"{req.width}*{req.height}",
            "Num": req.num
        }
    }
    body_bytes = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    body_sha256 = hashlib.sha256(body_bytes).hexdigest()

    # 2. 构造规范请求串（仅签名计算使用换行，不会流入HTTP头）
    query_str = f"Action={ACTION}&Version={VERSION}"
    canonical_headers = (
        f"content-type:application/json; charset=utf-8\n"
        f"host:{HOST}\n"
        f"x-content-sha256:{body_sha256}\n"
        f"x-date:{x_date}\n"
    )
    signed_header_list = "content-type;host;x-content-sha256;x-date"
    canonical_request = "\n".join([
        "POST",
        "/",
        query_str,
        canonical_headers.rstrip("\n"),
        signed_header_list,
        body_sha256
    ])

    # 3. 生成单行无换行鉴权头
    auth_value = calc_authorization(ak, sk, x_date, date_stamp, canonical_request)
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "X-Content-Sha256": body_sha256,
        "X-Date": x_date,
        "Authorization": auth_value
    }

    # 4. 发起POST请求
    api_url = f"https://{HOST}?{query_str}"
    resp = requests.post(api_url, headers=headers, data=body_bytes, timeout=120)
    resp_json = resp.json()
    logger.info(f"火山接口完整返回日志: {resp_json}")

    # 捕获火山业务错误
    meta_info = resp_json.get("ResponseMetadata", {})
    err_info = meta_info.get("Error")
    if err_info:
        raise Exception(f"{err_info['Code']} | {err_info['Message']}")

    # 提取图片链接
    img_items = resp_json["Result"]["StableDiffusion"]["Images"]
    return [item["ImageUrl"] for item in img_items]

# -------------------------- 绘图POST业务接口 --------------------------
@app.post("/api/v1/image/generate")
def api_generate_image(body: ImageDrawReq):
    try:
        img_urls = jimeng_cvprocess_draw(body)
        return {"code": 0, "msg": "生成成功", "data": {"image_urls": img_urls}}
    except Exception as e:
        logger.error(f"绘图失败：{str(e)}")
        return {"code": -1, "detail": str(e)}

# -------------------------- 前端首页页面 --------------------------
@app.get("/", response_class=HTMLResponse)
def html_index():
    page_html = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>即梦AI绘图官方Demo</title>
<style>
*{margin:0;padding:0;box-sizing:border-box;font-family:system-ui}
.wrap{max-width:620px;margin:40px auto;padding:0 16px}
.card{border:1px solid #e5e7eb;border-radius:14px;padding:24px}
h2{margin-bottom:20px;color:#1f2937}
textarea,input{width:100%;padding:12px;border:1px solid #d1d5db;border-radius:8px;margin-bottom:14px;font-size:14px}
textarea{height:110px;resize:none}
.row{display:flex;gap:12px}
button{width:100%;padding:14px;background:#2563eb;color:#fff;border:none;border-radius:8px;font-size:16px;font-weight:500}
#result{margin-top:20px;white-space:pre-wrap;color:#dc2626;font-size:14px}
</style>
</head>
<body>
<div class="wrap">
<div class="card">
<h2>AI生图 | 即梦CVProcess官方对接</h2>
<textarea id="prompt" placeholder="正向提示词，描述画面内容"></textarea>
<textarea id="neg" placeholder="反向提示词，规避瑕疵"></textarea>
<div class="row">
<input type="number" id="w" value="832" placeholder="宽度">
<input type="number" id="h" value="1216" placeholder="高度">
</div>
<button onclick="submitDraw()">一键生成图片</button>
<div id="result"></div>
</div>
</div>
<script>
async function submitDraw(){
    const resBox = document.getElementById("result");
    resBox.innerText = "接口请求中，请稍候...";
    const payload = {
        prompt: document.getElementById("prompt").value,
        negative_prompt: document.getElementById("neg").value,
        width: Number(document.getElementById("w").value),
        height: Number(document.getElementById("h").value),
        num: 1
    };
    try{
        const fetchResp = await fetch("/api/v1/image/generate",{
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload)
        });
        const data = await fetchResp.json();
        if(data.code === 0){
            resBox.style.color = "#16a34a";
            resBox.innerText = "✅ 生成成功\n图片链接：\n" + data.data.image_urls.join("\n");
        }else{
            resBox.style.color = "#dc2626";
            resBox.innerText = "❌ 生成失败\n" + JSON.stringify(data, null, 2);
        }
    }catch(err){
        resBox.style.color = "#dc2626";
        resBox.innerText = "网络异常：" + err.toString();
    }
}
</script>
</body>
</html>
    """
    return page_html

# -------------------------- 本地启动入口（Render兼容） --------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))