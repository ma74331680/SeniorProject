import os, json, uuid, shutil, asyncio
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import websockets
import aiofiles

COMFY_HOST = os.getenv("COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.getenv("COMFY_PORT", "8188"))
COMFY_HTTP = f"http://{COMFY_HOST}:{COMFY_PORT}"
COMFY_WS   = f"ws://{COMFY_HOST}:{COMFY_PORT}/ws"

DATA_DIR = Path("./data")
UPLOAD_DIR = DATA_DIR / "uploads"
WORKFLOW_TEMPLATE = Path("../workflow/img2sticker.json")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Comfy Gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 生產環境請改成你的網域
    allow_methods=["*"],
    allow_headers=["*"],
)

def _safe(n:str)->str: return n.replace("/","_").replace("\\","_")

def build_prompt(face_name:str, pose_name:str)->dict:
    wf = json.loads(WORKFLOW_TEMPLATE.read_text(encoding="utf-8"))
    wf_str = json.dumps(wf).replace("{{FACE_IMAGE}}", face_name).replace("{{POSE_IMAGE}}", pose_name)
    return {"prompt": json.loads(wf_str)}

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
                    "detail": r.text,               # ← 這裡會是關鍵訊息
                    "payload_shape": type(payload["prompt"]).__name__,
                    "node_keys_example": list(payload["prompt"].keys())[:5],
                },
                status_code=502,
            )

        data = r.json()
        prompt_id = data.get("prompt_id")


@app.websocket("/ws/progress/{prompt_id}")
async def ws_progress(ws: WebSocket, prompt_id: str):
    await ws.accept()
    client_id = uuid.uuid4().hex
    try:
        async with websockets.connect(f"{COMFY_WS}?clientId={client_id}", ping_interval=None) as cws:
            while True:
                raw = await cws.recv()  # string
                # 只轉發含有該 prompt_id 的事件（簡化：直接轉）
                # 前端可自行檢查 JSON 裡的 prompt_id
                await ws.send_text(raw)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type":"error","message":str(e)})
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
            subfolder = img_meta.get("subfolder","")
            typ = img_meta.get("type","output")
        except Exception:
            return JSONResponse({"error":"no image output"}, status_code=404)

        # 代理回傳實際圖片位元流
        img = await client.get(f"{COMFY_HTTP}/view", params={
            "filename": filename, "subfolder": subfolder, "type": typ
        })
        img.raise_for_status()
        return StreamingResponse(img.iter_bytes(), media_type=img.headers.get("content-type","image/png"))
    
    r = await client.post(f"{COMFY_HTTP}/prompt", json=payload)
    if r.status_code >= 400:
        # 把 ComfyUI 的錯誤文字印出或直接回傳，方便你定位
        detail = r.text
        # 也把重點節點打印出來檢查
        return JSONResponse(
            {"error": "comfyui_prompt_failed", "detail": detail, "sample": list(payload["prompt"].keys())[:5]},
            status_code=502,
        )
    r.raise_for_status()
    data = r.json()