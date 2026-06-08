# Vibing-Cloud RunPod serverless worker (Hub build).
# Wraps Microsoft's VibeVoice vLLM ASR server behind a RunPod handler.
#
# Hub note: the RunPod Hub has NO network-volume option, so the ~16GB ASR
# checkpoint is BAKED into the image at build time (downloaded once here, then
# read from local disk on every cold start). This is what makes a scale-to-zero
# cold start representative instead of a 16GB re-download every time.
#
# Gemma polish is OFF by default for this first spike (gated Google model that
# needs an HF token at build). We measure the dominant ASR cold start first.
FROM vllm/vllm-openai:v0.14.1

# The vllm/vllm-openai base image has no git, so install it first (exit 127 otherwise).
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# Clone Microsoft's VibeVoice repo; the installed vLLM plugin is used at runtime.
RUN git clone --depth 1 https://github.com/microsoft/VibeVoice.git /app
WORKDIR /app

# Handler deps. transformers/pillow are for the (disabled) Gemma path.
RUN pip install --no-cache-dir \
      runpod \
      requests \
      "huggingface_hub[cli]" \
      hf_transfer \
      "transformers>=4.57" \
      pillow

# Install the VibeVoice package + vLLM extra at build time. The handler launches
# `vllm serve` directly, so runtime cold start is vLLM boot + local model load.
RUN pip install --no-cache-dir -e "/app[vllm]"

# Bake the ASR weights and generated tokenizer files into the image's HF cache
# (no network volume on the Hub).
ENV HF_HOME=/models/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1
RUN huggingface-cli download microsoft/VibeVoice-ASR
# Run the tokenizer tool by file path (it's a standalone stdlib script with no
# package context; there is no vllm_plugin/tools/__init__.py, so `-m` would fail).
RUN python3 -c "import subprocess, sys; from huggingface_hub import snapshot_download; model_path = snapshot_download('microsoft/VibeVoice-ASR', local_files_only=True); subprocess.check_call([sys.executable, '/app/vllm_plugin/tools/generate_tokenizer_files.py', '--output', model_path])"

# Runtime defaults. Gemma off; ASR gets the whole GPU for this spike.
ENV VIBING_GEMMA_ENABLED=0 \
    VIBING_ASR_GPU_MEM_UTIL=0.88 \
    VIBING_ASR_MAX_MODEL_LEN=256 \
    VIBING_ASR_MAX_NUM_SEQS=1 \
    VIBING_ASR_MAX_NUM_BATCHED_TOKENS=256 \
    VIBING_ASR_MAX_OUTPUT_TOKENS=128 \
    VIBING_ASR_ENFORCE_EAGER=1 \
    VIBING_ASR_LOCAL_FILES_ONLY=1 \
    VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COPY handler.py /app/handler.py

# RunPod serverless entrypoint (override the base image's vLLM entrypoint).
ENTRYPOINT []
CMD ["python3", "-u", "/app/handler.py"]
