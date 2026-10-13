"""REST API server for fast Whisper transcription."""

import os
import tempfile
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
from .transcriber import FastWhisperTranscriber

app = FastAPI()
_model: Optional[FastWhisperTranscriber] = None

class TranscriptionResponse(BaseModel):
    text: str
    duration: float

class BatchTranscriptionResponse(BaseModel):
    results: List[TranscriptionResponse]
    total_time: float

def get_model() -> FastWhisperTranscriber:
    global _model
    if _model is None:
        _model = FastWhisperTranscriber(
            draft_model=os.getenv("DRAFT_MODEL", "tiny"),
            target_model=os.getenv("TARGET_MODEL", "large-v3"),
            draft_k=int(os.getenv("DRAFT_K", "6")),
            top_p=float(os.getenv("TOP_P", "0.0")),
            device=os.getenv("DEVICE", None),
        )
    return _model

@app.get("/")
async def root(): return {"status": "ok", "message": "Whisper Speculative Decoding API"}

@app.get("/health")
async def health(): return {"status": "ok", "model_loaded": _model is not None}

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...), max_tokens: int = Query(128, ge=1, le=448)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        try:
            result = get_model().transcribe(tmp_path, max_tokens=max_tokens, return_timing=True)
            return TranscriptionResponse(text=result["texts"], duration=result["total_time"])
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe/batch", response_model=BatchTranscriptionResponse)
async def transcribe_batch(files: List[UploadFile] = File(...), max_tokens: int = Query(128, ge=1, le=448)):
    try:
        tmp_paths = []
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(await file.read())
                tmp_paths.append(tmp.name)
        
        try:
            result = get_model().transcribe(tmp_paths, max_tokens=max_tokens, return_timing=True)
            texts = result["texts"] if isinstance(result["texts"], list) else [result["texts"]]
            avg_time = result["total_time"] / len(texts)
            return BatchTranscriptionResponse(
                results=[TranscriptionResponse(text=t, duration=avg_time) for t in texts],
                total_time=result["total_time"]
            )
        finally:
            for path in tmp_paths: os.unlink(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
