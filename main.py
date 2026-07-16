import json
import time
import logging
import requests
from datetime import datetime
from typing import List
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# 导入外部配置、签名工具
from config import CONF, HOST, ACTION, VERSION, SERVICE, REGION, FIX_REQ_KEY
from volc_sign import build_auth_header

# 日志初始化
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="模库AI 即梦绘图服务")
# 绑定前端模板目录
templates = Jinja2Templates(directory="templates")

# 请求入参模型
class DrawRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 832
    height: int = 1216
    num: int = 1

# 底层绘图服务（内置3次过期自动重试）
def jimeng_draw_service(req: DrawRequest) -> List[str]:
    max_retry = 3
    for retry_count in range(max_retry):
        # 每次重试强制生成全新UTC时间戳
        now_utc = datetime.utcnow()
        x_date = now_utc.strftime("%Y%m%dT%H:%M:%SZ")
        date_short = now_utc.strftime("%Y%m%d")
        # 日期兜底校验
        if not date_short or len(date_short) != 8:
            date_short = "20260708"

        # 构造请求体
        body_dict = {
            "ReqKey": FIX_REQ_KEY,
            "StableDiffusion": {
                "Prompt": req.prompt,
                "NegativePrompt": req.negative_prompt,
                "ImageSize": f"{req.width}*{req.height}",
                "Num": req.num
            }
        }
        body_bytes = json.dumps(body_dict, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        body_sha256 = hashlib.sha256(body_bytes).hexdigest()

        query_str = f"Action={ACTION}&Version={VERSION}"
        canonical_header_lines = (
            f"content-type:application/json; charset=utf-8\n"
            f"host:{HOST}\n"
            f"x-content-sha256:{body_sha256}\n"
            f"x-date:{x_date}\n"
        )
        signed_header_keys = "content-type;host;x-content-sha256;x-date"
        canonical_request = "\n".join([
            "POST", "/", query_str, canonical_header_lines.rstrip("\n"), signed_header_keys, body_sha256
        ])

        # 调用外部签名函数生成鉴权头
        auth_header = build_auth_header(
            ak=CONF.ak,
            sk=CONF.sk,
            x_date=x_date,
            date_stamp=date_short,
            canonical_req=canonical_request,
            region=REGION,
            service=SERVICE
        )

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "X-Content-Sha256": body_sha256,
            "X-Date": x_date,
            "Authorization": auth_header
        }
        api_url = f"https://{HOST}?{query_str}"
        resp = requests.post(api_url, headers=headers, data=body_bytes, timeout=120)
        resp_json = resp.json()
        logger.info(f"火山接口完整返回日志：{resp_json}")

        meta = resp_json.get("ResponseMetadata", {})
        err = meta.get("Error")
        if err:
            # 仅时间过期自动重试，其余报错直接抛出
            if err["Code"] == "InvalidTimestamp" and retry_count < max_retry - 1:
                logger.warning(f"签名过期，第{retry_count+1}次重试...")
                time.sleep(1)
                continue
            raise Exception(f"{err['Code']} | {err['Message']}")

        image_list = resp_json["Result"]["StableDiffusion"]["Images"]
        return [item["ImageUrl"] for item in image_list]
    raise Exception("连续3次重试签名仍过期，请重启Render容器同步时钟")

# -------------------------- 全部API接口写在此文件 --------------------------
# 绘图生成接口（核心POST接口）
@app.post("/api/v1/image/generate")
def api_generate_image(body: DrawRequest):
    if not CONF.is_valid():
        return {"code": -1, "detail": 
"JIMENG_AK: AKLTNTM1ZGEyOTM0NWZmNDg1Yjk5ZjczMTk3YzA1ZWVlZGE/
JIMENG_SK: WkdVNU1qYzBZV1JsTVRKak5HUTRaRGd5WmpNeE1EY3pZMkk0Tmpnd1ptUQ== "}
    try:
        img_urls = jimeng_draw_service(body)
        return {"code": 0, "msg": "生成成功", "data": {"image_urls": img_urls}}
    except Exception as e:
        logger.error(f"绘图失败：{str(e)}")
        return {"code": -1, "detail": str(e)}

# 首页前端页面接口
@app.get("/", response_class=HTMLResponse)
def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 服务启动入口
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)