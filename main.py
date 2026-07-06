import time
import logging
import requests
import json
from typing import Optional, List, Dict, Any, Union
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

# 日志基础配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("super-ai-gateway")

# FastAPI实例
app = FastAPI(title="智聚AI | 超级聚合引擎", version="2.0")
app.mount("/static", StaticFiles(directory="."), name="static")

# 全局环境变量统一读取
import os
CONF = {
    # 文本对话模型
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
    },
    # 文心一格绘图
    "wenxin": {
        "api_key": os.getenv("WENXIN_API_KEY"),
        "secret_key": os.getenv("WENXIN_SECRET_KEY"),
        "token_url": os.getenv("WENXIN_TOKEN_URL"),
        "draw_url": os.getenv("WENXIN_DRAW_URL"),
    },
    # 字节即梦绘图
    "jimeng": {
        "ak": os.getenv("JIMENG_AK"),
        "sk": os.getenv("JIMENG_SK"),
        "region": os.getenv("JIMENG_REGION", "cn-beijing"),
        "req_key": os.getenv("JIMENG_REQ_KEY"),
        "endpoint": "https://visual.volcengineapi.com"
    }
}

# -------------------------- 请求数据模型 --------------------------
# 文本对话入参
class UnifiedChatReq(BaseModel):
    model_type: str
    prompt: str
    system_prompt: Optional[str] = Field(default="你是全能AI助手，回答通俗易懂、专业简洁")
    temperature: float = Field(default=0.7, ge=0, le=1)
    max_tokens: int = Field(default=1024, gt=0)

# AI绘图入参【兼容前端model_type、width、height自由输入尺寸】
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
    def transform_fields(cls, values):
        if values.get("model_type") and not values.get("draw_type"):
            values["draw_type"] = values["model_type"]
        w = values.get("width")
        h = values.get("height")
        if w and h and not values.get("size"):
            values["size"] = f"{w}x{h}"
        if not values.get("draw_type"):
            raise ValueError("必须传入 model_type / draw_type，仅支持 wenxin / jimeng")
        return values

# 统一返回格式
class UnifiedChatResp(BaseModel):
    code: int
    msg: str
    data: Dict[str, Any]

# -------------------------- 模型调度适配器 --------------------------
class ModelAdapter:
    # 豆包对话接口
    @staticmethod
    def call_doubao(req: UnifiedChatReq) -> str:
        cfg = CONF["doubao"]
        headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
        payload = {
            "model": cfg["model_id"],
            "messages": [{"role":"system","content":req.system_prompt},{"role":"user","content":req.prompt}],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens
        }
        resp = requests.post(cfg["endpoint"], headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            raise Exception(f"豆包调用异常:{resp.text}")
        return resp.json()["choices"][0]["message"]["content"]

    # DeepSeek对话接口
    @staticmethod
    def call_deepseek(req: UnifiedChatReq) -> str:
        cfg = CONF["deepseek"]
        headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
        payload = {
            "model": cfg["model"],
            "messages": [{"role":"system","content":req.system_prompt},{"role":"user","content":req.prompt}],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens
        }
        resp = requests.post(cfg["endpoint"], headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            raise Exception(f"DeepSeek调用异常:{resp.text}")
        return resp.json()["choices"][0]["message"]["content"]

    # 通义千问对话接口
    @staticmethod
    def call_qwen(req: UnifiedChatReq) -> str:
        cfg = CONF["qwen"]
        headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
        input_text = f"{req.system_prompt}\n用户：{req.prompt}"
        payload = {"model": cfg["model"], "input": {"messages": [{"role": "user", "content": input_text}]}, "parameters": {"temperature": req.temperature}}
        resp = requests.post(cfg["endpoint"], headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            raise Exception(f"通义千问调用异常:{resp.text}")
        return resp.json()["output"]["text"]

    # 文心一格token缓存
    _wenxin_token_cache = {"token": "", "expire_time": 0}
    @staticmethod
    def get_wenxin_access_token() -> str:
        cfg = CONF["wenxin"]
        now_ts = time.time()
        if ModelAdapter._wenxin_token_cache["token"] and ModelAdapter._wenxin_token_cache["expire_time"] > now_ts + 120:
            return ModelAdapter._wenxin_token_cache["token"]
        params = {"grant_type":"client_credentials","client_id":cfg["api_key"],"client_secret":cfg["secret_key"]}
        resp = requests.get(cfg["token_url"], params=params, timeout=30)
        res_data = resp.json()
        if "access_token" not in res_data:
            raise Exception("文心密钥错误，获取鉴权失败")
        ModelAdapter._wenxin_token_cache["token"] = res_data["access_token"]
        ModelAdapter._wenxin_token_cache["expire_time"] = now_ts + res_data["expires_in"]
        return ModelAdapter._wenxin_token_cache["token"]

    # 文心一格绘图
    @staticmethod
    def call_wenxin_draw(img_req: UnifiedImageReq) -> List[str]:
        token = ModelAdapter.get_wenxin_access_token()
        cfg = CONF["wenxin"]
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"model":"ernie-vilg-v2","prompt":img_req.prompt,"negative_prompt":img_req.negative_prompt,"size":img_req.size,"n":img_req.num}
        resp = requests.post(cfg["draw_url"], headers=headers, json=payload, timeout=90)
        json_data = resp.json()
        if json_data.get("error_code"):
            raise Exception(json_data.get("error_msg"))
        return [item["url"] for item in json_data["data"]]

    # 即梦AI V4签名【终极修复：无换行、无多余空格、region匹配、header小写】
    @staticmethod
    def call_jimeng_draw(img_req: UnifiedImageReq) -> List[str]:
        import hmac
        import hashlib
        import json
        from datetime import datetime

        server_utc = datetime.utcnow()
        x_date = server_utc.strftime("%Y%m%dT%H:%M:%SZ")
        date_short = server_utc.strftime("%Y%m%d")

        cfg = CONF["jimeng"]
        host = "visual.volcengineapi.com"
        action = "CVProcess"
        version = "2018-08-01"
        url = f"https://{host}?Action={action}&Version={version}"
        service = "cv"
        region = cfg.get("region", "cn-beijing")

        body = {
            "ReqKey": cfg["req_key"],
            "StableDiffusion": {
                "Prompt": img_req.prompt,
                "NegativePrompt": img_req.negative_prompt,
                "ImageSize": img_req.size,
                "Num": img_req.num
            }
        }
        body_json = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        payload_sha256 = hashlib.sha256(body_json.encode("utf-8")).hexdigest()

        method = "POST"
        uri = "/"
        query = f"Action={action}&Version={version}"
        headers_raw = "content-type:application/json; charset=utf-8\nhost:" + host + "\nx-content-sha256:" + payload_sha256 + "\nx-date:" + x_date + "\n"
        signed_headers = "content-type;host;x-content-sha256;x-date"
        canonical_req = method + "\n" + uri + "\n" + query + "\n" + headers_raw + "\n" + signed_headers + "\n" + payload_sha256

        cr_sha256 = hashlib.sha256(canonical_req.encode("utf-8")).hexdigest()
        scope = date_short + "/" + region + "/" + service + "/request"
        string_to_sign = "HMAC-SHA256\n" + x_date + "\n" + scope + "\n" + cr_sha256

        def hmac_sha256(key, msg):
            return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

        k_date = hmac_sha256(cfg["sk"].encode("utf-8"), date_short)
        k_region = hmac_sha256(k_date, region)
        k_service = hmac_sha256(k_region, service)
        k_signing = hmac_sha256(k_service, "request")
        sig = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

        # 核心修复：auth单行拼接，逗号前后完全无空格，杜绝空白字符报错
        auth = "HMAC-SHA256 Credential=" + cfg['ak'] + "/" + scope + ",SignedHeaders=" + signed_headers + ",Signature=" + sig

        headers = {
            "content-type": "application/json; charset=utf-8",
            "x-content-sha256": payload_sha256,
            "x-date": x_date,
            "Authorization": auth
        }

        resp = requests.post(url, headers=headers, data=body_json, timeout=120)
        res_json = resp.json()
        logger.info(f"【即梦接口完整返回日志】{res_json}")
        meta = res_json.get("ResponseMetadata", {})
        err = meta.get("Error")
        if err:
            raise Exception(f"即梦接口鉴权/生成失败：{err['Code']} {err['Message']}")
        img_list = [item["ImageUrl"] for item in res_json["Result"]["StableDiffusion"]["Images"]]
        return img_list

# -------------------------- 业务API接口 --------------------------
# 首页前端页面
@app.get("/")
async def index_page():
    return FileResponse("index.html")

# 单模型对话接口
@app.post("/api/v1/chat", response_model=UnifiedChatResp)
def unified_chat(body: UnifiedChatReq):
    start_time = time.time()
    try:
        dispatch_map = {
            "doubao":ModelAdapter.call_doubao,
            "deepseek":ModelAdapter.call_deepseek,
            "qwen":ModelAdapter.call_qwen
        }
        if body.model_type not in dispatch_map:
            raise HTTPException(status_code=400, detail="仅支持 doubao / deepseek / qwen")
        ans = dispatch_map[body.model_type](body)
        return UnifiedChatResp(code=200, msg="响应成功", data={
            "model":body.model_type,"answer":ans,"cost_ms":round((time.time()-start_time)*1000,2)
        })
    except Exception as e:
        logger.error("对话异常", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# 多模型并行对比接口
@app.post("/api/v1/chat_batch")
def chat_batch(prompt: str, system_prompt: str = "专业AI助手", temperature: float = 0.7):
    model_list = ["doubao", "deepseek", "qwen"]
    res = {}
    for m in model_list:
        try:
            req = UnifiedChatReq(model_type=m,prompt=prompt,system_prompt=system_prompt,temperature=temperature)
            if m=="doubao":
                ans=ModelAdapter.call_doubao(req)
            elif m=="deepseek":
                ans=ModelAdapter.call_deepseek(req)
            else:
                ans=ModelAdapter.call_qwen(req)
            res[m] = {"status":"ok","content":ans}
        except Exception as e:
            res[m] = {"status":"fail","err":str(e)}
    return {"code":200,"data":res}

# 统一绘图生成接口
@app.post("/api/v1/image/generate", response_model=UnifiedChatResp)
def image_generate(body: UnifiedImageReq):
    start_time = time.time()
    try:
        if body.draw_type == "wenxin":
            urls = ModelAdapter.call_wenxin_draw(body)
        elif body.draw_type == "jimeng":
            urls = ModelAdapter.call_jimeng_draw(body)
        else:
            raise HTTPException(status_code=400, detail="仅支持：wenxin 文心一格 / jimeng 即梦AI")
        return UnifiedChatResp(code=200, msg="绘图完成", data={
            "draw_model":body.draw_type,"image_urls":urls,"cost_ms":round((time.time()-start_time)*1000,2)
        })
    except Exception as e:
        logger.error("绘图接口全局异常", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# 服务健康检测
@app.get("/health")
def health():
    return {"code":200,"msg":"后端网关全速运行"}

# 服务启动入口（Render端口10000）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=False)