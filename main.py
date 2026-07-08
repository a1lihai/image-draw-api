import time
import logging
import requests
import json
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator
import os

# 日志初始化
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-gateway")

# FastAPI 实例
app = FastAPI(title="智聚AI绘图对话平台", version="2.0")
app.mount("/static", StaticFiles(directory="."), name="static")

# 全局配置读取
CONF = {
    "jimeng": {
        "ak": os.getenv("JIMENG_AK"),
        "sk": os.getenv("JIMENG_SK"),
        "region": os.getenv("JIMENG_REGION", "cn-beijing"),
        "req_key": os.getenv("JIMENG_REQ_KEY"),
    },
    "wenxin": {
        "api_key": os.getenv("WENXIN_API_KEY"),
        "secret_key": os.getenv("WENXIN_SECRET_KEY"),
        "token_url": os.getenv("WENXIN_TOKEN_URL"),
        "draw_url": os.getenv("WENXIN_DRAW_URL"),
    },
    "doubao": {
        "api_key": os.getenv("DOUBAO_API_KEY"),
        "endpoint": os.getenv("DOUBAO_ENDPOINT"),
        "model_id": os.getenv("DOUBAO_MODEL_ID"),
    },
    "deepseek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "endpoint": os.getenv("DEEPSEEK_ENDPOINT"),
        "model": os.getenv("DEEPSEEK_MODEL"),
    },
    "qwen": {
        "api_key": os.getenv("QWEN_API_KEY"),
        "endpoint": os.getenv("QWEN_ENDPOINT"),
        "model": os.getenv("QWEN_MODEL"),
    }
}

# 对话入参模型
class UnifiedChatReq(BaseModel):
    model_type: str
    prompt: str
    system_prompt: Optional[str] = Field(default="你是全能AI助手，回答简洁易懂")
    temperature: float = Field(default=0.7, ge=0, le=1)
    max_tokens: int = Field(default=1024, gt=0)

# 绘图入参：兼容前端 model_type / width / height
class UnifiedImageReq(BaseModel):
    model_type: Optional[str] = None
    draw_type: Optional[str] = None
    prompt: str
    negative_prompt: str = Field(default="模糊,低画质,畸形,水印,文字,丑脸,多余肢体")
    width: Optional[int] = None
    height: Optional[int] = None
    size: Optional[str] = None
    num: int = Field(default=1, ge=1, le=4)

    @model_validator(mode="before")
    def transform_input(cls, values):
        # 前端model_type映射后端draw_type
        if values.get("model_type") and not values.get("draw_type"):
            values["draw_type"] = values["model_type"]
        # 宽高自动拼接size
        w = values.get("width")
        h = values.get("height")
        if w and h and not values.get("size"):
            values["size"] = f"{w}x{h}"
        if not values.get("draw_type"):
            raise ValueError("必须传入 model_type，仅支持 wenxin / jimeng")
        return values

# 统一返回结构
class CommonResp(BaseModel):
    code: int
    msg: str
    data: Dict[str, Any]

# 模型调度类
class ModelAdapter:
    # 豆包对话
    @staticmethod
    def call_doubao(req: UnifiedChatReq) -> str:
        cfg = CONF["doubao"]
        headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
        payload = {
            "model": cfg["model_id"],
            "messages": [{"role": "system", "content": req.system_prompt}, {"role": "user", "content": req.prompt}],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens
        }
        res = requests.post(cfg["endpoint"], headers=headers, json=payload, timeout=60)
        if res.status_code != 200:
            raise Exception(f"豆包请求异常: {res.text}")
        return res.json()["choices"][0]["message"]["content"]

    # DeepSeek对话
    @staticmethod
    def call_deepseek(req: UnifiedChatReq) -> str:
        cfg = CONF["deepseek"]
        headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
        payload = {
            "model": cfg["model"],
            "messages": [{"role": "system", "content": req.system_prompt}, {"role": "user", "content": req.prompt}],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens
        }
        res = requests.post(cfg["endpoint"], headers=headers, json=payload, timeout=60)
        if res.status_code != 200:
            raise Exception(f"DeepSeek请求异常: {res.text}")
        return res.json()["choices"][0]["message"]["content"]

    # 通义千问对话
    @staticmethod
    def call_qwen(req: UnifiedChatReq) -> str:
        cfg = CONF["qwen"]
        headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
        full_prompt = f"{req.system_prompt}\n用户：{req.prompt}"
        payload = {
            "model": cfg["model"],
            "input": {"messages": [{"role": "user", "content": full_prompt}]},
            "parameters": {"temperature": req.temperature}
        }
        res = requests.post(cfg["endpoint"], headers=headers, json=payload, timeout=60)
        if res.status_code != 200:
            raise Exception(f"通义千问请求异常: {res.text}")
        return res.json()["output"]["text"]

    # 文心一格Token缓存
    _wenxin_token_cache = {"token": "", "expire": 0}
    @staticmethod
    def get_wenxin_token() -> str:
        cfg = CONF["wenxin"]
        now = time.time()
        cache = ModelAdapter._wenxin_token_cache
        if cache["token"] and cache["expire"] > now + 120:
            return cache["token"]
        params = {"grant_type": "client_credentials", "client_id": cfg["api_key"], "client_secret": cfg["secret_key"]}
        res = requests.get(cfg["token_url"], params=params, timeout=30)
        data = res.json()
        if "access_token" not in data:
            raise Exception("文心密钥无效，获取Token失败")
        cache["token"] = data["access_token"]
        cache["expire"] = now + data["expires_in"]
        return cache["token"]

    # 文心一格绘图
    @staticmethod
    def call_wenxin_draw(img_req: UnifiedImageReq) -> List[str]:
        token = ModelAdapter.get_wenxin_token()
        cfg = CONF["wenxin"]
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {
            "model": "ernie-vilg-v2",
            "prompt": img_req.prompt,
            "negative_prompt": img_req.negative_prompt,
            "size": img_req.size,
            "n": img_req.num
        }
        res = requests.post(cfg["draw_url"], headers=headers, json=payload, timeout=90)
        data = res.json()
        if data.get("error_code"):
            raise Exception(data["error_msg"])
        return [item["url"] for item in data["data"]]

    # 【最终稳定版即梦签名】无斜杠丢失、无空格、无换行、大小写统一
  import hmac
import hashlib
from datetime import datetime

def get_auth_header(ak, sk, region, service, date_short, x_date, canonical_request):
    k_date = hmac.new(sk.encode("utf-8"), date_short.encode("utf-8"), hashlib.sha256).digest()
    k_region = hmac.new(k_date, region.encode("utf-8"), hashlib.sha256).digest()
    k_service = hmac.new(k_region, service.encode("utf-8"), hashlib.sha256).digest()
    k_signing = hmac.new(k_service, b"request", hashlib.sha256).digest()

    cr_hash = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    string_to_sign = f"HMAC-SHA256\n{x_date}\n{date_short}/{region}/{service}/request\n{cr_hash}"
    signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"HMAC-SHA256 Credential={ak.strip('/')}/{date_short}/{region}/{service}/request,SignedHeaders=content-type;host;x-content-sha256;x-date,Signature={signature}"

@staticmethod
def call_jimeng_draw(img_req: UnifiedImageReq) -> List[str]:
    import json
    cfg = CONF["jimeng"]
    host = "visual.volcengineapi.com"
    url = f"https://{host}?Action=CVProcess&Version=2022-08-31"
    req_key = "jimeng_high_aes_general_v21_L"
    region = "cn-north-1"
    service = "cv"

    now = datetime.utcnow()
    x_date = now.strftime("%Y%m%dT%H:%M:%SZ")
    date_short = now.strftime("%Y%m%d") or "20260708"

    body = {
        "ReqKey": req_key,
        "StableDiffusion": {
            "Prompt": img_req.prompt,
            "NegativePrompt": img_req.negative_prompt,
            "ImageSize": img_req.size,
            "Num": img_req.num
        }
    }
    body_bin = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    body_sha256 = hashlib.sha256(body_bin).hexdigest()

    canonical_headers = f"content-type:application/json; charset=utf-8\nhost:{host}\nx-content-sha256:{body_sha256}\nx-date:{x_date}\n"
    signed_headers = "content-type;host;x-content-sha256;x-date"
    canonical_req = f"POST\n/\nAction=CVProcess&Version=2022-08-31\n{canonical_headers}\n{signed_headers}\n{body_sha256}"

    auth = get_auth_header(cfg["ak"], cfg["sk"], region, service, date_short, x_date, canonical_req)
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "X-Content-Sha256": body_sha256,
        "X-Date": x_date,
        "Authorization": auth
    }
    resp = requests.post(url, headers=headers, data=body_bin, timeout=120)
    resp_data = resp.json()
    logger.info(resp_data)
    err = resp_data.get("ResponseMetadata", {}).get("Error")
    if err:
        raise Exception(f"{err['Code']}: {err['Message']}")
    return [item["ImageUrl"] for item in resp_data["Result"]["StableDiffusion"]["Images"]]

# 对话接口
@app.post("/api/v1/chat", response_model=CommonResp)
def chat_endpoint(body: UnifiedChatReq):
    start = time.time()
    dispatch_map = {
        "doubao": ModelAdapter.call_doubao,
        "deepseek": ModelAdapter.call_deepseek,
        "qwen": ModelAdapter.call_qwen
    }
    if body.model_type not in dispatch_map:
        raise HTTPException(status_code=400, detail="仅支持 doubao / deepseek / qwen")
    try:
        ans = dispatch_map[body.model_type](body)
        return CommonResp(
            code=200,
            msg="对话成功",
            data={
                "model": body.model_type,
                "answer": ans,
                "cost_ms": round((time.time() - start) * 1000, 2)
            }
        )
    except Exception as e:
        logger.error("对话接口异常", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# 绘图主接口
@app.post("/api/v1/image/generate", response_model=CommonResp)
def draw_endpoint(body: UnifiedImageReq):
    start = time.time()
    try:
        if body.draw_type == "wenxin":
            urls = ModelAdapter.call_wenxin_draw(body)
        elif body.draw_type == "jimeng":
            urls = ModelAdapter.call_jimeng_draw(body)
        else:
            raise HTTPException(status_code=400, detail="仅支持 wenxin 文心一格 / jimeng 即梦AI")
        return CommonResp(
            code=200,
            msg="绘图完成",
            data={
                "draw_model": body.draw_type,
                "image_urls": urls,
                "cost_ms": round((time.time() - start) * 1000, 2)
            }
        )
    except Exception as e:
        logger.error("绘图全局异常", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# 健康检测
@app.get("/health")
def health_check():
    return {"code": 200, "msg": "服务正常运行"}

# 启动入口
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=False)