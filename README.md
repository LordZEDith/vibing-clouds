# Vibing-Cloud — RunPod serverless cold-start spike

Purpose: prove (or kill) the idea of running VibeVoice-ASR + Gemma-4 polish on a
**RunPod serverless, scale-to-zero** GPU instead of on the user's machine. The whole
point of this spike is to measure **cold start** before rewriting the desktop client.

## What's here

| file | what it does |
|------|--------------|
| `handler.py` | RunPod worker. Boots Microsoft's VibeVoice vLLM ASR server (`:8000`, OpenAI-compatible) + loads Gemma-4 via transformers. Each job: audio → transcript → polished text, with decomposed timings. |
| `Dockerfile` | `vllm/vllm-openai:v0.14.1` + the VibeVoice repo + handler deps. |
| `bench_coldstart.py` | Runs **locally**. Hits the endpoint cold then warm, prints the table + verdict. |

## Architecture (why it's shaped this way)

VibeVoice-ASR has a first-class vLLM path (Microsoft's `vllm_plugin`) that exposes an
**OpenAI-compatible `/v1/chat/completions`** endpoint — audio is sent as a base64
`audio_url`. Gemma-4-e2b is loaded in-process via `transformers` (the same code path
the desktop app already uses, with `device="cuda"`). Both share one 24GB GPU; ASR's
`--gpu-memory-utilization` is dropped to `0.55` (env `VIBING_ASR_GPU_MEM_UTIL`) to leave
room for Gemma. Microsoft serves ASR in **bf16 (unquantized)** → plan for a 24GB card
(L4 / A5000 / A10 / 4090).

## Deploy (GitHub build, no local Docker)

1. **Push this repo to GitHub.** RunPod's GitHub integration builds from a Dockerfile in
   the repo. Either make `runpod-handler/` the repo root, or set the build context/Dockerfile
   path to `runpod-handler/Dockerfile` in the RunPod template.

2. **Create a Network Volume** (RunPod → Storage). ~40GB is plenty for the ASR + Gemma
   weights. This is what makes cold start survivable — without it, every cold worker
   re-downloads ~16GB+ of weights.

3. **Create a Serverless Endpoint** → *New Endpoint* → source **GitHub repo** → pick this
   repo/branch. Set:
   - **GPU**: 24GB (L4 / A5000 / A10 / 4090).
   - **Network volume**: attach the one from step 2 → it mounts at `/runpod-volume`
     (the Dockerfile points `HF_HOME` there).
   - **FlashBoot**: ON. This caches the worker so scale-to-zero cold starts resume far
     faster than a true cold boot — measure with it on, it's the realistic case.
   - **Active workers**: `0` (we are testing pure scale-to-zero). Max workers: `1`.
   - **Idle timeout**: short (e.g. 5s) so you can force cold starts between runs.
   - **Container disk**: ≥20GB.

4. **Pre-cache the weights onto the volume (one time).** The very first worker downloads
   `microsoft/VibeVoice-ASR` (~16GB) + `google/gemma-4-E2B-it` to `/runpod-volume/hf`.
   Trigger one request (next step) and just expect the first-ever cold start to be long
   (download-bound). Every cold start after that reads weights off the volume — that's the
   number we actually care about.

## Run the benchmark

```bash
cd runpod-handler
pip install requests
export RUNPOD_API_KEY=...        # Settings → API Keys
export RUNPOD_ENDPOINT_ID=...    # the endpoint's id
python3 bench_coldstart.py --idle-wait 360 --warm 10
```

`--idle-wait 360` waits 6 min first so the worker scales to zero and the next request is
a genuine cold start. Drop it to `0` to test warm-only.

## Reading the result

- **warm p50 < ~2.5s** → great, feels local.
- **cold < ~25s** → livable behind a "warming up…" HUD on first dictation.
- **cold > ~45s** → scale-to-zero loses; reconsider 1 always-on active worker (costs
  ~24/7 GPU) or a smaller/quantized ASR checkpoint.

The handler reports `asr_ready_s` vs `gemma_load_s` separately so you can see whether
cold start is dominated by vLLM engine init, weight load, or Gemma — each has a different
fix.

## Verify points (spike assumptions to confirm on first deploy)

- `start_server.py` path and its `--gpu-memory-utilization` / `--port` flags match the
  current VibeVoice `main`. If Microsoft moved the script, set `VIBING_ASR_START_SCRIPT`.
- Gemma loader class (`AutoModelForImageTextToText`) matches the `gemma-4-E2B-it` repo.
- Two engines (vLLM ASR + transformers Gemma) actually co-fit in 24GB at GMU 0.55 — if
  OOM, lower `VIBING_ASR_GPU_MEM_UTIL` or set `VIBING_GEMMA_ENABLED=0` to isolate ASR
  cold start first.
