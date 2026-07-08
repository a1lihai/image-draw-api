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
    @staticmethod
    def call_jimeng_draw(img_req: UnifiedImageReq) -> List[str]:
        import hmac
        import hashlib
        import json
        from datetime import datetime

        # UTC标准时间戳
        now_utc = datetime.utcnow()
        x_date = now_utc.strftime("%Y%m%dT%H:%M:%SZ")
        date_short = now_utc.strftime("%Y%m%d")

        cfg = CONF["jimeng"]
        host = "visual.volcengineapi.com"
        # 官方固定必填参数，硬写死杜绝丢失Action/Version
        action = "CVProcess"
        version = "2022-08-31"
        api_url = f"https://{host}?Action={action}&Version={version}"
        service_name = "cv"
        region = "cn-north-1"

        # 官方固定ReqKey，无需控制台创建应用
        fixed_req_key = "jimeng_high_aes_general_v21_L"

        # 请求Body
        req_body = {
            "ReqKey": fixed_req_key,
            "StableDiffusion": {
            "Prompt": img_req.prompt,
            "NegativePrompt": img_req.negative_prompt,
            "ImageSize": img_req.size,
            "Num": img_req.num
          }
        }
        body_raw = json.dumps(req_body, separators=(",", ":"), ensure_ascii=False)
        body_sha256 = hashlib.sha256(body_raw.encode("utf-8")).hexdigest()

        http_method = "POST"
        uri_path = "/"
        query_str = f"Action={action}&Version={version}"
        signed_header_keys = "content-type;host;x-content-sha256;x-date"

        # 规范签名头，硬编码换行，无隐形空格
        canonical_header_lines = (
            f"content-type:application/json; charset=utf-8\n"
            f"host:{host}\n"
            f"x-content-sha256:{body_sha256}\n"
            f"x-date:{x_date}\n"
        )
        canonical_request = "\n".join([
            http_method,
            uri_path,
            query_str,
            canonical_header_lines.rstrip("\n"),
            signed_header_keys,
            body_sha256
        ])
        cr_sha = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
        # scope强制分段，AK和region之间固定斜杠，绝不会丢失
        scope = f"{date_short}/{region}/{service_name}/request"
        string_to_sign = f"HMAC-SHA256\n{x_date}\n{scope}\n{cr_sha}"

        # HMAC签名推导函数
        def hmac_256(key_bytes, msg):
            return hmac.new(key_bytes, msg.encode("utf-8"), hashlib.sha256).digest()

        k_date = hmac_256(cfg["sk"].encode("utf-8"), date_short)
        k_region = hmac_256(k_date, region)
        k_service = hmac_256(k_region, service_name)
        k_sign = hmac_256(k_service, "request")
        final_signature = hmac.new(k_sign, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

        # 分段拼接鉴权串，逗号前后零空格，彻底解决空白字符报错
        credential_segment = f"Credential={cfg['ak']}/{scope}"
        signed_header_segment = f"SignedHeaders={signed_header_keys}"
        signature_segment = f"Signature={final_signature}"
        auth_header_value = f"HMAC-SHA256  {credential_segment},{signed_header_segment},{signature_segment}"

        headers = {
            "content-type": "application/json; charset=utf-8",
            "x-content-sha256": body_sha256,
            "x-date": x_date,
            "Authorization": auth_header_value
        }

        resp = requests.post(api_url, headers=headers, data=body_raw, timeout=120)
        resp_data = resp.json()
        logger.info(f"即梦接口完整返回日志: {resp_data}")
        meta = resp_data.get("ResponseMetadata", {})
        err_info = meta.get("Error")
        if err_info:
            raise Exception(f"火山API鉴权/生成失败：{err_info['Code']} - {err_info['Message']}")

        img_url_list = [item["ImageUrl"] for item in resp_data["Result"]["StableDiffusion"]["Images"]]
        return img_url_list
# 首页前端页面
@app.get("/")
async def index():
    return FileResponse("index.html")

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