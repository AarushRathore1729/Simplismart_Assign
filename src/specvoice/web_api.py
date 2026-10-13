"""Single-GPU upload API using the verified SpecVoice inference engine."""

from __future__ import annotations

import argparse
import os
from contextlib import asynccontextmanager
from io import BytesIO
from threading import Lock
from typing import Annotated, Literal

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from .asr.transcriber import SpeculativeWhisperTranscriber

MAX_UPLOAD_BYTES = 32 * 1024 * 1024


def create_app(engine=None):
    lock = Lock()

    @asynccontextmanager
    async def lifespan(application):
        application.state.engine = engine or SpeculativeWhisperTranscriber.from_pretrained(
            target_model_name=os.getenv("TARGET_MODEL", "openai/whisper-large-v3"),
            draft_model_name=os.getenv("DRAFT_MODEL", "openai/whisper-tiny"),
        )
        yield

    application = FastAPI(title="Fast Whisper", version="0.3.1", lifespan=lifespan)

    @application.get("/health")
    def health():
        model = getattr(application.state, "engine", None)
        return {"ready": model is not None, "device": str(model.device) if model else None}

    @application.post("/transcribe")
    def transcribe(
        file: Annotated[UploadFile, File()],
        mode: Literal["baseline", "speculative", "optimized", "adaptive"] = "optimized",
        max_new_tokens: Annotated[int, Query(ge=1, le=256)] = 128,
    ):
        payload = file.file.read(MAX_UPLOAD_BYTES + 1)
        if not payload:
            raise HTTPException(422, "Upload a non-empty audio file")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Maximum upload size is 32 MiB")
        try:
            audio, rate = sf.read(BytesIO(payload), dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1, dtype=np.float32)
            # FastAPI executes sync routes in its thread pool. Serializing the
            # shared GPU engine prevents concurrent requests from overlapping
            # inference, timing or memory measurements.
            with lock:
                result = application.state.engine.transcribe_array(
                    audio, rate, mode=mode, max_new_tokens=max_new_tokens
                )
            return result.to_dict()
        except (ValueError, sf.LibsndfileError) as error:
            raise HTTPException(422, str(error)) from error

    return application


app = create_app()


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port, workers=1)


if __name__ == "__main__":
    main()
