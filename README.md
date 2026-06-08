[![Runpod](https://api.runpod.io/badge/LordZEDith/vibing-clouds)](https://console.runpod.io/hub/LordZEDith/vibing-clouds)

# Vibing-Cloud — RunPod serverless cold-start spike

Purpose: prove (or kill) the idea of running VibeVoice-ASR on a **RunPod serverless,
scale-to-zero** GPU instead of on the user's machine. The whole point of this spike is to
measure **cold start** before rewriting the Vibing desktop client.

This first cut is **ASR only** — it measures the dominant cold-start cost (the ~16GB
VibeVoice checkpoint + vLLM boot). Gemma-4 transcript polish is wired in `handler.py` but
**off by default** (`VIBING_GEMMA_ENABLED=0`): it's a gated Google model needing an HF
token at build, so it's added once the ASR number is in.

## What's here

| file | what it does |
|------|--------------|
| `handler.py` | RunPod worker. Boots VibeVoice through vLLM (`:8000`, OpenAI-compatible). Each job: base64 audio → transcript, with decomposed timings. Optional Gemma polish when enabled. |
| `Dockerfile` | `vllm/vllm-openai:v0.14.1` + the VibeVoice repo, with the ASR weights and tokenizer files **baked in**. |
| `.runpod/hub.json` | RunPod Hub deploy config (audio category, 24GB GPU pool, env toggles). |
| `.runpod/tests.json` | Hub smoke test — a real base64 WAV through the ASR path. |
| `bench_coldstart.py` | Runs **locally**. Hits the endpoint cold then warm, prints the table + verdict. |

## Architecture (why it's shaped this way)

VibeVoice-ASR has a first-class vLLM path (Microsoft's `vllm_plugin`) exposing an
**OpenAI-compatible `/v1/chat/completions`** endpoint — audio sent as a base64 `audio_url`.
Microsoft serves it in **bf16 (unquantized)** → a 24GB card (L4 / A5000 / A10 / 4090).
The handler launches `vllm serve` directly instead of `start_server.py` so worker cold
start does not redo package installation, model download checks, or tokenizer generation.

The RunPod **Hub has no network-volume option**, so the weights are **baked into the
image** (`huggingface-cli download` at build). They download once during the Hub build, not
on every cold start — which is what makes the cold-start number representative.

## Deploy via the RunPod Hub

The Hub builds from a tagged GitHub **release** of this repo. Steps the Hub walks you through:

1. `.runpod/hub.json` + `.runpod/tests.json` — ✅ in this repo.
2. `Dockerfile` + `handler.py` — ✅ in this repo.
3. **Add the badge** (top of this README) — ✅.
4. **Create a GitHub release** → the Hub builds the image, runs `tests.json`, and publishes
   the listing. The build downloads ~16GB, so expect it to take a while.
5. From the Hub listing, **Deploy** → choose the **ASR only** preset, GPU 24GB, FlashBoot
   **on**, active workers `0`, max `1`, short idle timeout.

## Run the benchmark

```bash
pip install requests
export RUNPOD_API_KEY=...        # Settings → API Keys
export RUNPOD_ENDPOINT_ID=...    # the deployed endpoint's id
python3 bench_coldstart.py --idle-wait 360 --warm 10
```

`--idle-wait 360` waits 6 min so the worker scales to zero and the next request is a genuine
cold start. Drop it to `0` for warm-only.

## Reading the result

- **warm p50 < ~2.5s** → great, feels local.
- **cold < ~25s** → livable behind a "warming up…" HUD on first dictation.
- **cold > ~45s** → scale-to-zero loses; reconsider 1 always-on active worker (24/7 GPU
  cost) or a smaller/quantized ASR checkpoint.

The handler reports `asr_ready_s` separately so you can see how much of cold start is vLLM
engine init vs weight load.

## Verify points (assumptions to confirm on first build)

- `microsoft/VibeVoice-ASR` is the right, ungated model id and downloads at build.
- L4 startup should keep `VIBING_ASR_MAX_MODEL_LEN=1024`,
  `VIBING_ASR_MAX_NUM_BATCHED_TOKENS=1024`, `VIBING_ASR_MAX_NUM_SEQS=1`, and
  `VIBING_ASR_ENFORCE_EAGER=1`. Microsoft's long-form defaults are `65536` tokens and
  `64` sequences, which are not appropriate for a single short dictation request.
- Keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; `PYTORCH_ALLOC_CONF` does not
  configure the CUDA allocator and can leave enough reserved memory unavailable to trip
  the VibeVoice audio encoder on L4.
- The direct `vllm serve` options in `handler.py` should continue matching Microsoft's
  VibeVoice vLLM launcher if their plugin changes.
- To add Gemma later: set `VIBING_GEMMA_ENABLED=1`, drop `VIBING_ASR_GPU_MEM_UTIL` to ~0.55,
  bake the Gemma weights with an HF token, and use a 24GB+ (ideally 48GB) card.
