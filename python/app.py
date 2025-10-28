import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Optional

import aiofiles
import httpx
import websockets
from fastapi import (FastAPI, File, Form, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

COMFY_HOST = os.getenv("COMFY_HOST", "120.110.113.166")
COMFY_PORT = int(os.getenv("COMFY_PORT", "8188"))
COMFY_HTTP = f"http://{COMFY_HOST}:{COMFY_PORT}"
COMFY_WS = f"ws://{COMFY_HOST}:{COMFY_PORT}/ws"

DATA_DIR = Path("./data")
UPLOAD_DIR = DATA_DIR / "uploads"
WORKFLOW_TEMPLATE = Path("../workflow/img2sticker.json")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Comfy Gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生產環境請改成你的網域
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe(n: str) -> str:
    return n.replace("/", "_").replace("\\", "_")


def build_prompt(face_name: str, pose_name: str) -> dict:
    wf = json.loads(WORKFLOW_TEMPLATE.read_text(encoding="utf-8"))
    # 替換檔名
    wf_str = (
        json.dumps(wf)
        .replace("{{FACE_IMAGE}}", face_name)
        .replace("{{POSE_IMAGE}}", pose_name)
    )
    prompt = json.loads(wf_str)

    # 規則：class_type 是 SaveImage（或 SaveImageWebsocket），且 filename_prefix/subfolder 含 "final"
    final_ids = []
    for node_id, node in prompt.items():
        ctype = node.get("class_type", "")
        if ctype in ("SaveImage", "SaveImageWebsocket"):
            inputs = node.get("inputs", {})
            prefix = (inputs.get("filename_prefix") or "").lower()
            subf = (inputs.get("subfolder") or "").lower()
            if "final" in prefix or "final" in subf:
                final_ids.append(node_id)

    # 沒標記就退而求其次：取全部 SaveImage 的最後一個當「成品」
    if not final_ids:
        for node_id, node in prompt.items():
            if node.get("class_type", "") in ("SaveImage", "SaveImageWebsocket"):
                final_ids.append(node_id)
        final_ids = final_ids[-1:]  # 只留最後一個

    return {"prompt": prompt, "final_ids": final_ids}


@app.post("/api/jobs")
async def create_job(
    face: UploadFile = File(...),
    pose: UploadFile = File(...),
    seed: Optional[int] = Form(None),
    prompt_text: Optional[str] = Form(None),
):
    # 存到本機（可選：也會上傳到 ComfyUI /upload/image）
    job_id = uuid.uuid4().hex
    face_name = _safe(f"{job_id}_face_{face.filename}")
    pose_name = _safe(f"{job_id}_pose_{pose.filename}")
    face_path = UPLOAD_DIR / face_name
    pose_path = UPLOAD_DIR / pose_name

    async with aiofiles.open(face_path, "wb") as f:
        while chunk := await face.read(1024 * 1024):
            await f.write(chunk)
    async with aiofiles.open(pose_path, "wb") as f:
        while chunk := await pose.read(1024 * 1024):
            await f.write(chunk)

    async with httpx.AsyncClient(timeout=120) as client:
        # 上傳到 ComfyUI 的 input 資料夾（不同主機時很方便；同主機也可直接讀 input 目錄）
        for p in (face_path, pose_path):
            try:
                files = {"image": (p.name, p.open("rb"), "image/png")}
                await client.post(f"{COMFY_HTTP}/upload/image", files=files)
            except Exception:
                # 若你不想用 /upload/image，也可把 UPLOAD_DIR 指向 ComfyUI/input 路徑
                pass

        payload = build_prompt(face_name, pose_name)

        # 建議加 client_id（方便之後 WS 過濾事件）
        client_id = uuid.uuid4().hex
        payload_with_client = {"prompt": payload["prompt"], "client_id": client_id}

        r = await client.post(f"{COMFY_HTTP}/prompt", json=payload_with_client)

        if r.status_code >= 400:
            # 把 ComfyUI 的錯誤原文拋回前端，方便定位
            return JSONResponse(
                {
                    "error": "comfyui_prompt_failed",
                    "status_code": r.status_code,
                    "detail": r.text,  # ← 這裡會是關鍵訊息
                    "payload_shape": type(payload["prompt"]).__name__,
                    "node_keys_example": list(payload["prompt"].keys())[:5],
                },
                status_code=502,
            )

        data = r.json()
        prompt_id = data.get("prompt_id")

        if not prompt_id:
            return JSONResponse({"error": "no_prompt_id_from_comfy"}, status_code=502)

        return JSONResponse(
            {
                "prompt_id": prompt_id,
                "client_id": client_id,  # 一並回給前端，之後 WS 要用同一個
                "job_id": job_id,
                "face": face_name,
                "pose": pose_name,
                "final_node_ids": payload.get("final_ids", []),  # ★ 回給前端
            },
            status_code=200,
        )


@app.websocket("/ws/progress/{prompt_id}/{client_id}")
async def ws_progress(ws: WebSocket, prompt_id: str, client_id: str):
    await ws.accept()
    try:
        async with websockets.connect(
            f"{COMFY_WS}?clientId={client_id}", ping_interval=None
        ) as cws:
            while True:
                raw = await cws.recv()  # string
                # 只轉發含有該 prompt_id 的事件（簡化：直接轉）
                # 前端可自行檢查 JSON 裡的 prompt_id
                await ws.send_text(raw)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


@app.get("/api/result/{prompt_id}")
async def get_result(prompt_id: str):
    # 查詢歷史，取得輸出檔名，直接 proxy /view 回前端
    async with httpx.AsyncClient(timeout=120) as client:
        hist = await client.get(f"{COMFY_HTTP}/history/{prompt_id}")
        hist.raise_for_status()
        h = hist.json()
        # 取第一張輸出圖（你可依你的 SaveImage 命名/子資料夾來挑）
        # 通常在 h[prompt_id]["outputs"]["<SaveImage節點ID>"]["images"][0]
        try:
            first_node = next(iter(h[prompt_id]["outputs"].values()))
            img_meta = first_node["images"][0]
            filename = img_meta["filename"]
            subfolder = img_meta.get("subfolder", "")
            typ = img_meta.get("type", "output")
        except Exception:
            return JSONResponse({"error": "no image output"}, status_code=404)

        # 代理回傳實際圖片位元流
        img = await client.get(
            f"{COMFY_HTTP}/view",
            params={"filename": filename, "subfolder": subfolder, "type": typ},
        )
        img.raise_for_status()
        return StreamingResponse(
            img.iter_bytes(), media_type=img.headers.get("content-type", "image/png")
        )


async def _try_fetch_image_by_history(
    client: httpx.AsyncClient, prompt_id: str, final_node_id: Optional[str] = None
):
    r = await client.get(f"{COMFY_HTTP}/history/{prompt_id}")
    if r.status_code != 200:
        return None
    h = r.json()
    try:
        outputs = h[prompt_id]["outputs"]  # dict: node_id -> { "images":[...] }
        # 若有指定 final_node_id：只看它
        if (
            final_node_id
            and final_node_id in outputs
            and outputs[final_node_id].get("images")
        ):
            img_meta = outputs[final_node_id]["images"][-1]  # 取該節點最後一張
            return (
                img_meta["filename"],
                img_meta.get("subfolder", ""),
                img_meta.get("type", "output"),
            )

        # 否則：優先挑「名稱/子資料夾帶 final 的節點」，再 fallback 到所有節點的「最後一張」
        # 1) 篩節點 key 名稱含 "final"（有些人會把節點命成 SaveImage_final）
        for node_id, node_out in outputs.items():
            if "final" in str(node_id).lower():
                imgs = node_out.get("images") or []
                if imgs:
                    img_meta = imgs[-1]
                    return (
                        img_meta["filename"],
                        img_meta.get("subfolder", ""),
                        img_meta.get("type", "output"),
                    )

        # 2) 全部節點中挑「子資料夾/檔名前綴含 final」者
        for node_out in outputs.values():
            imgs = node_out.get("images") or []
            if not imgs:
                continue
            last = imgs[-1]
            if "final" in (
                last.get("subfolder", "").lower() + last.get("filename", "").lower()
            ):
                return (
                    last["filename"],
                    last.get("subfolder", ""),
                    last.get("type", "output"),
                )

        # 3) 完全沒有標記 → 取「所有節點中最後一個的最後一張」（最保守）
        last_node = next(reversed(outputs.values()))
        imgs = last_node.get("images") or []
        if imgs:
            last = imgs[-1]
            return (
                last["filename"],
                last.get("subfolder", ""),
                last.get("type", "output"),
            )
    except Exception:
        return None


def _looks_finished_ws(msg: dict, prompt_id: str) -> bool:
    try:
        t = msg.get("type")
        data = msg.get("data") or {}
        if data.get("prompt_id") == prompt_id and t in {
            "executed",
            "finished",
            "execution_end",
        }:
            return True
        if t == "status":
            status = data.get("status") or {}
            exec_info = status.get("exec_info") or {}
            if exec_info.get("prompt_id") == prompt_id and exec_info.get(
                "queue_remaining"
            ) in (0, "0"):
                return True
    except Exception:
        pass
    return False


@app.get("/api/result/{prompt_id}/wait")
async def wait_and_stream_result(
    prompt_id: str,
    client_id: str,
    final_node_id: Optional[str] = None,
    timeout: int = 420,
):
    """
    等 ComfyUI 完成該 prompt 後直接回傳圖片。
    用法：
      GET /api/result/<prompt_id>/wait?client_id=<同一個>&timeout=420
    """
    deadline = asyncio.get_event_loop().time() + timeout
    backoff = 0.6  # 起始輪詢間隔

    async with httpx.AsyncClient(timeout=30) as client:
        # 若已經出圖（萬一 history 先好了）
        hit = await _try_fetch_image_by_history(
            client, prompt_id, final_node_id=final_node_id
        )
        if hit:
            filename, subfolder, typ = hit
            img = await client.get(
                f"{COMFY_HTTP}/view",
                params={"filename": filename, "subfolder": subfolder, "type": typ},
            )
            img.raise_for_status()
            return StreamingResponse(
                img.aiter_bytes(),
                media_type=img.headers.get("content-type", "image/png"),
            )

        async def ws_waiter():
            uri = f"{COMFY_WS}?clientId={client_id}"
            async with websockets.connect(uri, ping_interval=None) as cws:
                while True:
                    raw = await asyncio.wait_for(
                        cws.recv(),
                        timeout=max(1, int(deadline - asyncio.get_event_loop().time())),
                    )
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    # 僅當 final 那顆 SaveImage executed / finished 才去抓圖
                    t = msg.get("type")
                    data = msg.get("data") or {}
                    if data.get("prompt_id") == prompt_id and t in {
                        "executed",
                        "finished",
                        "execution_end",
                    }:
                        # 若指定 final_node_id，需 data.node == final_node_id
                        if final_node_id is None or data.get("node") == final_node_id:
                            got = await _try_fetch_image_by_history(
                                client, prompt_id, final_node_id=final_node_id
                            )
                            if got:
                                return got

        async def poll_waiter():
            nonlocal backoff
            while True:
                if asyncio.get_event_loop().time() > deadline:
                    return None
                got = await _try_fetch_image_by_history(
                    client, prompt_id, final_node_id=final_node_id
                )
                if got:
                    return got
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 3.0)

        done, pending = await asyncio.wait(
            {asyncio.create_task(ws_waiter()), asyncio.create_task(poll_waiter())},
            return_when=asyncio.FIRST_COMPLETED,
            timeout=max(1, int(deadline - asyncio.get_event_loop().time())),
        )
        for t in pending:
            t.cancel()

        if not done:
            return JSONResponse(
                {"error": "timeout_waiting_image", "prompt_id": prompt_id},
                status_code=408,
            )

        hit = list(done)[0].result()
        if not hit:
            return JSONResponse(
                {"error": "timeout_waiting_image", "prompt_id": prompt_id},
                status_code=408,
            )

        filename, subfolder, typ = hit
        img = await client.get(
            f"{COMFY_HTTP}/view",
            params={"filename": filename, "subfolder": subfolder, "type": typ},
        )
        img.raise_for_status()
        return StreamingResponse(
            img.aiter_bytes(), media_type=img.headers.get("content-type", "image/png")
        )
