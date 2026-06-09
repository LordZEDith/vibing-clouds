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
| `Dockerfile` | `vllm/vllm-openai:v0.14.1` + the VibeVoice repo, with generated tokenizer files only. ASR weights come from RunPod cached models. |
| `.runpod/hub.json` | RunPod Hub deploy config (audio category, A100 80GB pool, env toggles). |
| `.runpod/tests.json` | Hub smoke test — a real base64 WAV through the ASR path. |
| `bench_coldstart.py` | Runs **locally**. Hits the endpoint cold then warm, prints the table + verdict. |

## Architecture (why it's shaped this way)

VibeVoice-ASR has a first-class vLLM path (Microsoft's `vllm_plugin`) exposing an
**OpenAI-compatible `/v1/chat/completions`** endpoint — audio sent as a base64 `audio_url`.
Microsoft serves it in **bf16 (unquantized)** → a 24GB card (L4 / A5000 / A10 / 4090).
The handler launches `vllm serve` directly instead of `start_server.py` so worker cold
start does not redo package installation, model download checks, or tokenizer generation.

The ASR weights are not baked into the image anymore. Use RunPod's **Model** field to cache
`microsoft/VibeVoice-ASR`; the worker resolves the cached Hugging Face snapshot at
`/runpod-volume/huggingface-cache/hub/` and passes that local path to vLLM. The image only
carries the smaller VibeVoice tokenizer files generated at build time.

## Deploy via the RunPod Hub

The Hub builds from a tagged GitHub **release** of this repo. Steps the Hub walks you through:

1. `.runpod/hub.json` + `.runpod/tests.json` — ✅ in this repo.
2. `Dockerfile` + `handler.py` — ✅ in this repo.
3. **Add the badge** (top of this README) — ✅.
4. **Create a GitHub release** → the Hub builds the image, runs `tests.json`, and publishes
   the listing. RunPod indexes GitHub releases, not plain commits; their docs say updates
   are usually indexed within an hour.
5. From the Hub listing, **Deploy** → choose the **ASR only** preset, GPU A100 80GB
   (`AMPERE_80`), FlashBoot **on**, active workers `0`, max `1`, short idle timeout.
6. In RunPod's endpoint configuration, scroll to **Model** and enter:

   ```text
   microsoft/VibeVoice-ASR
   ```

   Leave `VIBING_ASR_ALLOW_RUNTIME_DOWNLOAD=0`. The worker also forces Hugging Face offline
   mode in this cached-only path. If the Model field is missing or wrong, it fails fast
   instead of silently doing a slow model download.

## Run the benchmark

```bash
pip install requests
export RUNPOD_API_KEY=...        # Settings → API Keys
export RUNPOD_ENDPOINT_ID=...    # the deployed endpoint's id
python3 bench_coldstart.py --idle-wait 360 --warm 10
```

To benchmark a real recording:

```bash
python3 bench_coldstart.py \
  --audio /Users/vsha/Downloads/testrecording.mp3 \
  --hotwords "RPN,Rhys,Viktor,Adrian" \
  --warm 3
```

`--idle-wait 360` waits 6 min so the worker scales to zero and the next request is a genuine
cold start. Drop it to `0` for warm-only.

## Reading the result

- **warm p50 < ~2.5s** → great, feels local.
- **cold < ~25s** → livable behind a "warming up..." HUD on first dictation.
- **cold > ~45s** → scale-to-zero loses; compare RunPod cached models against Modal/other
  cold-start platforms before paying for an always-on worker.

The handler reports `asr_ready_s` separately so you can see how much of cold start is vLLM
engine init vs weight load.

## Verify points (assumptions to confirm on first build)

- `microsoft/VibeVoice-ASR` is the right, ungated model id and must be entered in the
  RunPod endpoint **Model** field for cached-model deployment.
- GPU pool selection is controlled by `.runpod/hub.json` at `config.gpuIds`, not by an
  environment variable in the Configure modal. The default is cost/performance balanced:
  `AMPERE_80`. This targets A100 80GB and excludes 24GB/48GB pools by omission.
  To exclude a specific GPU type within an included pool, prefix the GPU type with `-`
  (for example, `AMPERE_80,-NVIDIA A100-SXM4-80GB`).
- A100 deploy defaults should keep `VIBING_ASR_GPU_MEM_UTIL=0.88`,
  `VIBING_ASR_MAX_MODEL_LEN=2048`, `VIBING_ASR_MAX_NUM_BATCHED_TOKENS=2048`,
  `VIBING_ASR_MAX_NUM_SEQS=1`, and `VIBING_ASR_ENFORCE_EAGER=1`. Microsoft's long-form defaults are `65536` tokens and
  `64` sequences, which are not appropriate for a single short dictation request.
- Keep `VIBING_ASR_MAX_OUTPUT_TOKENS=1024` for real dictation. `128` truncated the
  structured ASR response in endpoint testing.
- Keep `RUNPOD_INIT_TIMEOUT=800`; RunPod can mark workers unhealthy if initialization
  exceeds 7 minutes, and this leaves enough room for slow first builds/tests.
- Keep both `PYTORCH_ALLOC_CONF=expandable_segments:True` and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; the warning name differs across
  PyTorch/vLLM builds.
- The direct `vllm serve` options in `handler.py` should continue matching Microsoft's
  VibeVoice vLLM launcher if their plugin changes.
- To add Gemma later: set `VIBING_GEMMA_ENABLED=1`, drop `VIBING_ASR_GPU_MEM_UTIL` to ~0.55,
  bake the Gemma weights with an HF token, and use a 24GB+ (ideally 48GB) card.
