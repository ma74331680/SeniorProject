# app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import requests
import time
from io import BytesIO

COMFY = "http://127.0.0.1:8188"   # 你的 ComfyUI 服務位址
TIMEOUT_SEC = 60                  # 最長等圖時間
POLL_INTERVAL = 0.8               # 輪詢間隔

app = FastAPI()

def submit_prompt(prompt_json: dict) -> str:
    # prompt_json 需包含 "client_id" 與 "prompt"（建議用 UI 的 Save (API format) 版）
    r = requests.post(f"{COMFY}/prompt", json=prompt_json, timeout=30)
    if r.status_code != 200:
        raise HTTPException(502, f"ComfyUI /prompt 失敗: {r.text}")
    # 回傳格式通常包含 prompt_id
    data = r.json()
    prompt_id = data.get("prompt_id") or data.get("promptId") or data.get("promptid")
    if not prompt_id:
        raise HTTPException(502, f"無法取得 prompt_id，回應: {data}")
    return prompt_id

def wait_images(prompt_id: str):
    """輪詢 history，直到有圖片清單或超時。回傳 [(filename, subfolder, type)]"""
    deadline = time.time() + TIMEOUT_SEC
    url = f"{COMFY}/history/{prompt_id}"
    while time.time() < deadline:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200 and resp.content:
            h = resp.json()
            # ComfyUI 的 history 結構：{ "<prompt_id>": { "outputs": { "<node_id>": { "images": [...] }}}}
            node_map = h.get(prompt_id, {}).get("outputs", {})
            images = []
            for node_id, out in node_map.items():
                for img in out.get("images", []) or []:
                    images.append((img["filename"], img.get("subfolder", ""), img.get("type", "output")))
            if images:
                return images
        time.sleep(POLL_INTERVAL)
    return []

def fetch_image(filename: str, subfolder: str, ftype: str) -> bytes:
    params = {"filename": filename, "subfolder": subfolder or "", "type": ftype or "output"}
    r = requests.get(f"{COMFY}/view", params=params, timeout=30)
    if r.status_code != 200:
        raise HTTPException(502, f"抓圖失敗: {r.text}")
    return r.content

@app.post("/generate-and-return-image")
def generate_and_return_image(payload: dict):
    """
    payload 直接放你要送給 ComfyUI 的 JSON（建議為 UI 匯出的 Save(API format)）
    例如：
    {
      "client_id": "abc123",
      "prompt": { ...workflow nodes... }
    }
    """
    prompt_id = submit_prompt(payload)
    images = wait_images(prompt_id)
    if not images:
        # 也可改回傳 JSON 告知「尚未產生」或附上 prompt_id 讓前端稍後再取
        raise HTTPException(504, "等待逾時，未取得圖片")

    # 這裡只取第一張；若要全部，自己包 zip 或回傳 URL 陣列
    filename, subfolder, ftype = images[0]
    img_bytes = fetch_image(filename, subfolder, ftype)

    # 推斷副檔名（ComfyUI 多半為 png/webp）
    media_type = "image/png" if filename.lower().endswith(".png") else "image/webp"
    return StreamingResponse(BytesIO(img_bytes), media_type=media_type,
                             headers={"Content-Disposition": f'inline; filename="{filename}"'})
