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
import re
import subprocess
import time
import urllib.request

import requests

# ---------------------------------------------------------------------------
# Config (all overridable from the RunPod template env)
# ---------------------------------------------------------------------------
ASR_MODEL = os.environ.get("VIBING_ASR_MODEL", "microsoft/VibeVoice-ASR")
HF_CACHE_ROOT = os.environ.get(
    "VIBING_HF_CACHE_ROOT", "/runpod-volume/huggingface-cache/hub"
)
ASR_PORT = int(os.environ.get("VIBING_ASR_PORT", "8000"))
ASR_BASE_URL = f"http://127.0.0.1:{ASR_PORT}"
# Leave enough VRAM for VibeVoice's multimodal startup profile.
ASR_GPU_MEM_UTIL = os.environ.get("VIBING_ASR_GPU_MEM_UTIL", "0.88")
# Microsoft's launcher defaults to 65536 tokens and 64 sequences for long-form,
# high-throughput ASR. Vibing sends one short dictation clip at a time, but real
# transcripts need more than the 256-token smoke-test cap.
ASR_MAX_MODEL_LEN = os.environ.get("VIBING_ASR_MAX_MODEL_LEN", "2048")
ASR_MAX_NUM_SEQS = os.environ.get("VIBING_ASR_MAX_NUM_SEQS", "1")
ASR_MAX_NUM_BATCHED_TOKENS = os.environ.get(
    "VIBING_ASR_MAX_NUM_BATCHED_TOKENS", ASR_MAX_MODEL_LEN
)
ASR_MAX_OUTPUT_TOKENS = int(os.environ.get("VIBING_ASR_MAX_OUTPUT_TOKENS", "1024"))
ASR_ENFORCE_EAGER = os.environ.get("VIBING_ASR_ENFORCE_EAGER", "1") == "1"
ASR_LOCAL_FILES_ONLY = os.environ.get("VIBING_ASR_LOCAL_FILES_ONLY", "1") == "1"
ASR_ALLOW_RUNTIME_DOWNLOAD = (
    os.environ.get("VIBING_ASR_ALLOW_RUNTIME_DOWNLOAD", "0") == "1"
)
ASR_TOKENIZER_PATH = os.environ.get(
    "VIBING_ASR_TOKENIZER_PATH", "/app/vibevoice-tokenizer"
)
ASR_BOOT_TIMEOUT_S = int(os.environ.get("VIBING_ASR_BOOT_TIMEOUT_S", "780"))

GEMMA_ENABLED = os.environ.get("VIBING_GEMMA_ENABLED", "1") == "1"
GEMMA_MODEL = os.environ.get("VIBING_GEMMA_MODEL", "google/gemma-4-E2B-it")
GEMMA_MAX_NEW_TOKENS = int(os.environ.get("VIBING_GEMMA_MAX_NEW_TOKENS", "512"))

if not ASR_ALLOW_RUNTIME_DOWNLOAD:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_WORKER_START = time.monotonic()
_COLD = True  # flipped False after the first job a worker serves
_GEMMA = {"processor": None, "model": None}
_INIT = {"asr_ready_s": None, "gemma_load_s": 0.0, "worker_init_s": None, "error": None}


def _log(msg: str) -> None:
    print(f"[handler +{time.monotonic() - _WORKER_START:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Worker init (this is the cold-start cost we are measuring)
# ---------------------------------------------------------------------------
_ASR_PROC: subprocess.Popen | None = None
_ASR_LOG = "/tmp/asr_server.log"


def _asr_log_tail(limit: int = 16000) -> str:
    # The vLLM EngineCore subprocess prints the *root cause* well before the API
    # server's re-raise, so capture a wide tail and strip ANSI colour codes.
    try:
        with open(_ASR_LOG, "rb") as f:
            raw = f.read()[-limit:].decode("utf-8", "replace")
        return re.sub(r"\x1b\[[0-9;]*m", "", raw)
    except Exception:
        return "(no asr server log)"


def _resolve_asr_model_path() -> str:
    if os.path.isdir(ASR_MODEL):
        _log(f"using ASR model directory from VIBING_ASR_MODEL: {ASR_MODEL}")
        return ASR_MODEL

    cached_path = _resolve_hf_cached_snapshot(ASR_MODEL)
    if cached_path:
        _log(f"using RunPod cached ASR model snapshot: {cached_path}")
        return cached_path

    if not ASR_ALLOW_RUNTIME_DOWNLOAD and ASR_LOCAL_FILES_ONLY:
        raise RuntimeError(
            f"RunPod cached model not found for {ASR_MODEL!r} under {HF_CACHE_ROOT}. "
            "Set the endpoint Model field to microsoft/VibeVoice-ASR, or set "
            "VIBING_ASR_ALLOW_RUNTIME_DOWNLOAD=1 for a one-off fallback download."
        )

    from huggingface_hub import snapshot_download

    _log(f"cached ASR model not found; falling back to snapshot_download({ASR_MODEL})")
    return snapshot_download(
        ASR_MODEL,
        local_files_only=ASR_LOCAL_FILES_ONLY and not ASR_ALLOW_RUNTIME_DOWNLOAD,
    )


def _resolve_hf_cached_snapshot(model_id: str) -> str | None:
    if "/" not in model_id:
        return None
    org, name = model_id.split("/", 1)
    model_root = os.path.join(HF_CACHE_ROOT, f"models--{org}--{name}")
    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")

    if os.path.isfile(refs_main):
        with open(refs_main, "r", encoding="utf-8") as f:
            snapshot_hash = f.read().strip()
        candidate = os.path.join(snapshots_dir, snapshot_hash)
        if os.path.isdir(candidate):
            return candidate

    if os.path.isdir(snapshots_dir):
        versions = [
            d
            for d in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, d))
        ]
        if versions:
            versions.sort()
            return os.path.join(snapshots_dir, versions[0])

    return None


def _start_asr_server() -> None:
    global _ASR_PROC
    model_path = _resolve_asr_model_path()
    cmd = [
        "vllm",
        "serve",
        model_path,
        "--served-model-name",
        "vibevoice",
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--max-num-seqs",
        ASR_MAX_NUM_SEQS,
        "--max-num-batched-tokens",
        ASR_MAX_NUM_BATCHED_TOKENS,
        "--max-model-len",
        ASR_MAX_MODEL_LEN,
        "--gpu-memory-utilization",
        ASR_GPU_MEM_UTIL,
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--chat-template-content-format",
        "openai",
        "--tensor-parallel-size",
        "1",
        "--data-parallel-size",
        "1",
        "--allowed-local-media-path",
        "/app",
        "--port",
        str(ASR_PORT),
    ]
    if os.path.isdir(ASR_TOKENIZER_PATH):
        cmd.extend(["--tokenizer", ASR_TOKENIZER_PATH])
    else:
        _log(f"ASR tokenizer path not found, using model tokenizer: {ASR_TOKENIZER_PATH}")
    if ASR_ENFORCE_EAGER:
        cmd.append("--enforce-eager")
    _log(f"launching ASR vLLM server: {' '.join(cmd)}")
    logf = open(_ASR_LOG, "wb")
    _ASR_PROC = subprocess.Popen(
        cmd, cwd="/app", env=os.environ.copy(), stdout=logf, stderr=subprocess.STDOUT
    )


def _wait_for_asr() -> None:
    deadline = time.monotonic() + ASR_BOOT_TIMEOUT_S
    health = f"{ASR_BASE_URL}/health"
    while time.monotonic() < deadline:
        # Fail fast (with the real error) if vLLM died instead of
        # blindly waiting out the whole timeout.
        if _ASR_PROC is not None and _ASR_PROC.poll() is not None:
            raise RuntimeError(
                f"ASR server exited early (code={_ASR_PROC.returncode}).\n"
                f"--- ASR server log tail ---\n{_asr_log_tail()}"
            )
        try:
            with urllib.request.urlopen(health, timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(
        f"ASR server not healthy after {ASR_BOOT_TIMEOUT_S}s.\n"
        f"--- ASR server log tail ---\n{_asr_log_tail()}"
    )


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
        "max_tokens": ASR_MAX_OUTPUT_TOKENS,
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
