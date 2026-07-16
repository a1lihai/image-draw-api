import hmac
import hashlib

def get_signing_key(secret_key: str, date_stamp: str, region: str, service: str):
    k_date = hmac.new(secret_key.encode("utf-8"), date_stamp.encode("utf-8"), hashlib.sha256).digest()
    k_region = hmac.new(k_date, region.encode("utf-8"), hashlib.sha256).digest()
    k_service = hmac.new(k_region, service.encode("utf-8"), hashlib.sha256).digest()
    k_signing = hmac.new(k_service, b"request", hashlib.sha256).digest()
    return k_signing

def build_auth_header(ak: str, sk: str, x_date: str, date_stamp: str, canonical_req: str, region: str, service: str):
    cr_hash = hashlib.sha256(canonical_req.encode("utf-8")).hexdigest()
    credential_scope = f"{date_stamp}/{region}/{service}/request"
    string_to_sign = f"HMAC-SHA256\n{x_date}\n{credential_scope}\n{cr_hash}"
    signing_key = get_signing_key(sk, date_stamp, region, service)
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    # 单行鉴权头，无换行、无多余空格
    auth = (
        f"HMAC-SHA256 "
        f"Credential={ak}/{credential_scope},"
        f"SignedHeaders=content-type;host;x-content-sha256;x-date,"
        f"Signature={signature}"
    )
    return auth