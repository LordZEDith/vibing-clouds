#!/usr/bin/env python3
"""Cold-start benchmark for the Vibing-Cloud RunPod serverless endpoint.

Answers the one question that decides the whole project: is scale-to-zero
serverless fast enough for a dictation app, or does cold start ruin it?

Usage:
    export RUNPOD_API_KEY=...        # RunPod account API key
    export RUNPOD_ENDPOINT_ID=...    # serverless endpoint id
    python3 bench_coldstart.py [--idle-wait 360] [--warm 10]

Procedure:
    1. (optional) wait --idle-wait seconds so the endpoint scales to zero,
       guaranteeing the next request hits a genuinely cold worker.
    2. Fire one request  -> COLD number (includes worker boot + model load).
    3. Fire --warm requests back to back -> WARM p50/p95.
    4. Print a table and a verdict.

The handler also reports its own decomposed timings (asr_ready_s, gemma_load_s),
so you can see how much of cold start is vLLM boot vs weight load vs queue.
"""
import argparse
import base64
import io
import mimetypes
import os
import statistics
import struct
import sys
import time
import wave

import requests

API_KEY = os.environ.get("RUNPOD_API_KEY")
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")


def make_sample_wav(seconds: float = 3.0, freq: float = 220.0, rate: int = 16000) -> str:
    """A short sine tone. Real ASR output is irrelevant for timing; we only need
    a valid audio payload the server will actually decode and run."""
    n = int(seconds * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(n):
            # cheap integer sine approximation without math.sin imports per-sample
            import math

            val = int(12000 * math.sin(2 * math.pi * freq * (i / rate)))
            frames += struct.pack("<h", val)
        w.writeframes(bytes(frames))
    return base64.b64encode(buf.getvalue()).decode()


def call(audio_b64: str, audio_mime: str, hotwords: list[str]) -> tuple[float, dict]:
    """One synchronous request via RunPod's /runsync. Returns (wall_seconds, output)."""
    base = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    body = {
        "input": {
            "audio_b64": audio_b64,
            "audio_mime": audio_mime,
            "hotwords": hotwords,
        }
    }
    t0 = time.monotonic()
    resp = requests.post(f"{base}/runsync", json=body, headers=headers, timeout=900)
    resp.raise_for_status()
    data = resp.json()
    status = data.get("status")
    output = data.get("output")
    job_id = data.get("id")
    while output is None and status in {"IN_QUEUE", "IN_PROGRESS"} and job_id:
        time.sleep(2)
        poll = requests.get(f"{base}/status/{job_id}", headers=headers, timeout=30)
        poll.raise_for_status()
        data = poll.json()
        status = data.get("status")
        output = data.get("output")
    wall = time.monotonic() - t0
    return wall, output or data


def load_audio_b64(path: str | None) -> tuple[str, str, str]:
    if not path:
        return make_sample_wav(), "audio/wav", "generated 3s WAV tone"
    audio_path = os.path.expanduser(path)
    mime = mimetypes.guess_type(audio_path)[0] or "application/octet-stream"
    with open(audio_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return encoded, mime, audio_path


def main() -> int:
    if not API_KEY or not ENDPOINT_ID:
        print("Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID env vars.", file=sys.stderr)
        return 2

    ap = argparse.ArgumentParser()
    ap.add_argument("--idle-wait", type=int, default=0,
                    help="seconds to wait first so the endpoint scales to zero (cold)")
    ap.add_argument("--warm", type=int, default=10, help="number of warm requests")
    ap.add_argument("--audio", help="audio file to send instead of a generated WAV tone")
    ap.add_argument(
        "--hotwords",
        default="Vibing",
        help="comma-separated hotwords to pass to the endpoint",
    )
    args = ap.parse_args()

    hotwords = [word.strip() for word in args.hotwords.split(",") if word.strip()]
    audio_b64, audio_mime, audio_label = load_audio_b64(args.audio)
    print(f"audio: {audio_label} ({audio_mime})")
    print(f"hotwords: {', '.join(hotwords) or '(none)'}")

    if args.idle_wait:
        print(f"waiting {args.idle_wait}s for endpoint to scale to zero...")
        time.sleep(args.idle_wait)

    print("\n=== COLD request ===")
    cold_wall, out = call(audio_b64, audio_mime, hotwords)
    timings = out.get("timings", {})
    print(f"  wall: {cold_wall:6.2f}s   handler says: {timings}")
    if out.get("error"):
        print(f"  ERROR from worker: {out['error']}")
        return 1
    if not timings.get("cold", True):
        print("  NOTE: worker reported cold=False — it was already warm. "
              "Re-run with --idle-wait to force a true cold start.")

    print(f"\n=== {args.warm} WARM requests ===")
    warms = []
    for i in range(args.warm):
        wall, out = call(audio_b64, audio_mime, hotwords)
        warms.append(wall)
        print(f"  [{i+1:2d}] {wall:5.2f}s  asr={out.get('timings', {}).get('asr_s')}  "
              f"polish={out.get('timings', {}).get('polish_s')}")

    print("\n=== RESULT ===")
    print(f"  cold first-token:   {cold_wall:6.2f}s")
    print(f"  worker boot/init:   {timings.get('worker_init_s')}s "
          f"(asr_ready={timings.get('asr_ready_s')}s, gemma_load={timings.get('gemma_load_s')}s)")
    if warms:
        p50 = statistics.median(warms)
        p95 = sorted(warms)[max(0, int(len(warms) * 0.95) - 1)]
        print(f"  warm p50:           {p50:6.2f}s")
        print(f"  warm p95:           {p95:6.2f}s")
        print(f"  cold penalty:       {cold_wall - p50:6.2f}s extra on the first dictation")
    else:
        print("  warm p50/p95:       skipped (--warm 0)")
    print("\nVerdict guide: warm p50 < 2.5s feels great. Cold < 25s is livable behind "
          "a 'warming up' HUD. Cold > 45s -> reconsider a min warm worker / FlashBoot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
