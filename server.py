from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from faster_whisper import WhisperModel
from collections import defaultdict, deque
import time
import requests
import tempfile
import os




app = FastAPI()

# ---- CONFIG ----
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b") 

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")  # tiny, base, small, medium, large-v3
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "auto")  # "auto" usually picks best on Mac
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8")  # good speed; try "float16" if you want
# Keep last 12 messages per session (user/assistant turns)
HISTORY = defaultdict(lambda: deque(maxlen=12))


_whisper = None

def get_whisper():
    global _whisper
    if _whisper is None:
        _whisper = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE,
        )
    return _whisper


class ChatReq(BaseModel):
    text: str
    session_id: str = "pi"


@app.post("/stt")
async def stt(audio: UploadFile = File(...)):
    """
    Receives WAV audio. Returns {"text": "..."} using faster-whisper.
    """
    data = await audio.read()

    # Write upload to a temp wav file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        model = get_whisper()

        segments, info = model.transcribe(
            tmp_path,
            language="en",          # change or remove for auto-detect
            vad_filter=True,        # helps with silence/noise
        )

        text = "".join(seg.text for seg in segments).strip()
        if not text:
            text = "(no speech detected)"
        return {"text": text}
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.post("/chat")
def chat(req: ChatReq):
    session = req.session_id

    # Add user message to history
    HISTORY[session].append({"role": "user", "content": req.text})

    # Build a compact prompt with history
    history_text = ""
    for m in HISTORY[session]:
        role = "User" if m["role"] == "user" else "Assistant"
        history_text += f"{role}: {m['content']}\n"

    prompt = (
        "You are a helpful, concise voice assistant. "
        "Use the conversation context. "
        "Reply in 1-3 short sentences unless the user asks for detail.\n\n"
        f"{history_text}"
        "Assistant:"
    )

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                # helpful knobs:
                "options": {
                    "temperature": 0.4,
                    "num_predict": 200
                }
            },
            timeout=120,
        )
        r.raise_for_status()
        out = r.json()
        reply = (out.get("response") or "").strip() or "I didn't get a response."

        # Add assistant reply to history
        HISTORY[session].append({"role": "assistant", "content": reply})
        return {"reply": reply}
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        reply = "LLM error. Check Ollama is running and model name is correct."
        HISTORY[session].append({"role": "assistant", "content": reply})
        return {"reply": f"{reply} Details: {err}"}
