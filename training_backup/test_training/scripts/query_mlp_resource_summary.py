#!/usr/bin/env python3
"""Query BytePlus/Volc MLP resource group and queue summaries without printing secrets."""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


HOST = "open.volcengineapi.com"
REGION = "ap-southeast-1"
SERVICE = "ml_platform"
VERSION = "2024-07-01"
RESOURCE_GROUP_ID = "r-20260128180700-np4h5"
RESOURCE_QUEUE_ID = "q-20260128181205-z9l2h"


def hmac_sha256(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def sign_key(secret: str, date: str) -> bytes:
    key = hmac_sha256(secret.encode("utf-8"), date)
    key = hmac_sha256(key, REGION)
    key = hmac_sha256(key, SERVICE)
    return hmac_sha256(key, "request")


def call(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    access_key = os.environ.get("HuoShanAccessKeyID") or os.environ.get("VOLC_ACCESSKEY")
    secret_key = os.environ.get("HuoShanSecretAccessKey") or os.environ.get("VOLC_SECRETKEY")
    if not access_key or not secret_key:
        raise RuntimeError("Missing HuoShanAccessKeyID/HuoShanSecretAccessKey in environment")

    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    body_hash = hashlib.sha256(body).hexdigest()
    now = dt.datetime.now(dt.timezone.utc)
    x_date = now.strftime("%Y%m%dT%H%M%SZ")
    short_date = now.strftime("%Y%m%d")

    query = urllib.parse.urlencode({"Action": action, "Version": VERSION})
    signed_headers = "content-type;host;x-date"
    canonical_headers = (
        "content-type:application/json\n"
        f"host:{HOST}\n"
        f"x-date:{x_date}\n"
    )
    canonical_request = "\n".join(
        ["POST", "/", query, canonical_headers, signed_headers, body_hash]
    )
    credential_scope = f"{short_date}/{REGION}/{SERVICE}/request"
    string_to_sign = "\n".join(
        [
            "HMAC-SHA256",
            x_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )
    signature = hmac.new(sign_key(secret_key, short_date), string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    authorization = (
        f"HMAC-SHA256 Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    req = urllib.request.Request(
        f"https://{HOST}/?{query}",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Host": HOST,
            "X-Date": x_date,
            "Authorization": authorization,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{action} HTTP {exc.code}: {text}") from exc


def unwrap(resp: dict[str, Any]) -> dict[str, Any]:
    if "Result" in resp and isinstance(resp["Result"], dict):
        return resp["Result"]
    return resp


def keep_resource_summary(item: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "Id",
        "Name",
        "Status",
        "ZoneIds",
        "ResourceCapability",
        "ResourceAllocated",
        "ResourceGroupId",
        "ResourceQueueId",
        "ResourceGroupID",
        "ResourceQueueID",
        "ResourceSpec",
        "ResourceSpecs",
        "ComputeResources",
    ]
    return {key: item[key] for key in keys if key in item}


def main() -> None:
    requests = [
        ("ListResourceGroups", {"Ids": [RESOURCE_GROUP_ID], "PageSize": 10, "PageNumber": 1}),
        ("ListResourceQueues", {"Ids": [RESOURCE_QUEUE_ID], "PageSize": 10, "PageNumber": 1}),
    ]
    output: dict[str, Any] = {}
    for action, payload in requests:
        try:
            result = unwrap(call(action, payload))
        except Exception as exc:
            output[action] = {"error": str(exc)}
            continue
        items = result.get("Items") or result.get("items") or []
        output[action] = {
            "total": result.get("TotalCount") or result.get("total"),
            "items": [keep_resource_summary(item) for item in items if isinstance(item, dict)],
        }

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise
