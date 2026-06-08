"""RunPod serverless handler — Vibing-Cloud cold-start spike.

Goal of this file: bring up BOTH inference paths inside one serverless worker and
report a decomposed cold-start number so we can decide whether scale-to-zero
serverless is viable for a dictation app.

Worker layout:
  - VibeVoice-ASR runs as Microsoft's vLLM OpenAI-compatible server on :8000
    (subprocess, launched once at worker init).
  - Gemma-4-e2b multimodal polish runs in-process via transformers (ports the
    app's existing polish_text_transformers_multimodal path, device -> cuda).

The handler is a thin proxy: audio -> ASR (/v1/chat/completions), then
{transcript + screenshot} -> Gemma polish.

Timings returned per job:
  cold              : True only for the first job a fresh worker serves
  worker_init_s     : ASR server boot + Gemma load, measured once at import
  asr_ready_s       : time for the vLLM ASR server to become healthy
  gemma_load_s      : time to load Gemma into VRAM (0 if GEMMA_ENABLED=0)
  asr_s / polish_s  : per-request inference time
"""

import base64
import io
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

import requests

# ---------------------------------------------------------------------------
# Config (all overridable from the RunPod template env)
# ---------------------------------------------------------------------------
ASR_MODEL = os.environ.get("VIBING_ASR_MODEL", "microsoft/VibeVoice-ASR")
ASR_PORT = int(os.environ.get("VIBING_ASR_PORT", "8000"))
ASR_BASE_URL = f"http://127.0.0.1:{ASR_PORT}"
# Leave VRAM headroom for Gemma when both share one GPU. Microsoft's default is 0.8.
ASR_GPU_MEM_UTIL = os.environ.get("VIBING_ASR_GPU_MEM_UTIL", "0.55")
ASR_START_SCRIPT = os.environ.get(
    "VIBING_ASR_START_SCRIPT", "/app/vllm_plugin/scripts/start_server.py"
)
ASR_BOOT_TIMEOUT_S = int(os.environ.get("VIBING_ASR_BOOT_TIMEOUT_S", "600"))

GEMMA_ENABLED = os.environ.get("VIBING_GEMMA_ENABLED", "1") == "1"
GEMMA_MODEL = os.environ.get("VIBING_GEMMA_MODEL", "google/gemma-4-E2B-it")
GEMMA_MAX_NEW_TOKENS = int(os.environ.get("VIBING_GEMMA_MAX_NEW_TOKENS", "512"))

_WORKER_START = time.monotonic()
_COLD = True  # flipped False after the first job a worker serves
_GEMMA = {"processor": None, "model": None}
_INIT = {"asr_ready_s": None, "gemma_load_s": 0.0, "worker_init_s": None, "error": None}


def _log(msg: str) -> None:
    print(f"[handler +{time.monotonic() - _WORKER_START:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Worker init (this is the cold-start cost we are measuring)
# ---------------------------------------------------------------------------
def _start_asr_server() -> None:
    cmd = [
        sys.executable,
        ASR_START_SCRIPT,
        "--gpu-memory-utilization",
        ASR_GPU_MEM_UTIL,
        "--port",
        str(ASR_PORT),
    ]
    _log(f"launching ASR vLLM server: {' '.join(cmd)}")
    # Inherit stdout/stderr so vLLM boot logs land in the RunPod worker log.
    subprocess.Popen(cmd, env=os.environ.copy())


def _wait_for_asr() -> None:
    deadline = time.monotonic() + ASR_BOOT_TIMEOUT_S
    health = f"{ASR_BASE_URL}/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"ASR server not healthy after {ASR_BOOT_TIMEOUT_S}s")


def _load_gemma() -> None:
    if not GEMMA_ENABLED:
        return
    t0 = time.monotonic()
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    _log(f"loading Gemma {GEMMA_MODEL} on cuda")
    _GEMMA["processor"] = AutoProcessor.from_pretrained(GEMMA_MODEL)
    _GEMMA["model"] = AutoModelForImageTextToText.from_pretrained(
        GEMMA_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    _INIT["gemma_load_s"] = round(time.monotonic() - t0, 2)
    _log(f"Gemma loaded in {_INIT['gemma_load_s']}s")


def _init_worker() -> None:
    t0 = time.monotonic()
    try:
        _start_asr_server()
        _load_gemma()  # loads while vLLM is still warming — overlaps cost
        _wait_for_asr()
        _INIT["asr_ready_s"] = round(time.monotonic() - t0, 2)
    except Exception as err:  # surfaced on every job so the bench sees it
        _INIT["error"] = repr(err)
        _log(f"INIT FAILED: {err!r}")
    finally:
        _INIT["worker_init_s"] = round(time.monotonic() - t0, 2)
        _log(f"worker init done in {_INIT['worker_init_s']}s ({_INIT})")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _transcribe(audio_b64: str, audio_mime: str, hotwords: list[str]) -> str:
    data_url = f"data:{audio_mime};base64,{audio_b64}"
    prompt = "Transcribe the audio."
    if hotwords:
        prompt += f" with extra info: {', '.join(hotwords)}"
    payload = {
        "model": "vibevoice",
        "messages": [
            {"role": "system", "content": "You transcribe audio accurately."},
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
        "stream": False,
    }
    resp = requests.post(
        f"{ASR_BASE_URL}/v1/chat/completions", json=payload, timeout=300
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _polish(transcript: str, screenshot_b64: str | None) -> str:
    if not GEMMA_ENABLED or _GEMMA["model"] is None:
        return transcript
    import torch
    from PIL import Image

    processor, model = _GEMMA["processor"], _GEMMA["model"]
    prompt = (
        "Clean up this dictation transcript into polished text. Use the on-screen "
        f"context if helpful. Return only the cleaned text.\n\nTranscript: {transcript}"
    )
    content: list[dict] = []
    if screenshot_b64:
        image = Image.open(io.BytesIO(base64.b64decode(screenshot_b64))).convert("RGB")
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True,
    ).to("cuda")
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=GEMMA_MAX_NEW_TOKENS, do_sample=False)
    return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# RunPod entrypoint
# ---------------------------------------------------------------------------
def handler(job: dict) -> dict:
    global _COLD
    cold = _COLD
    _COLD = False

    if _INIT["error"]:
        return {"error": f"worker init failed: {_INIT['error']}", "cold": cold}

    inp = job.get("input", {})
    audio_b64 = inp.get("audio_b64")
    if not audio_b64:
        return {"error": "missing input.audio_b64", "cold": cold}
    audio_mime = inp.get("audio_mime", "audio/wav")
    hotwords = inp.get("hotwords", [])
    screenshot_b64 = inp.get("screenshot_b64")

    t0 = time.monotonic()
    transcript = _transcribe(audio_b64, audio_mime, hotwords)
    t1 = time.monotonic()
    polished = _polish(transcript, screenshot_b64)
    t2 = time.monotonic()

    return {
        "rawTranscript": transcript,
        "polishedText": polished,
        "timings": {
            "cold": cold,
            "worker_uptime_s": round(time.monotonic() - _WORKER_START, 2),
            "worker_init_s": _INIT["worker_init_s"],
            "asr_ready_s": _INIT["asr_ready_s"],
            "gemma_load_s": _INIT["gemma_load_s"],
            "asr_s": round(t1 - t0, 2),
            "polish_s": round(t2 - t1, 2),
        },
    }


# Boot models at import so the first job measures a realistic cold start.
_init_worker()

if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
