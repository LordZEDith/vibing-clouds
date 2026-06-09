# Vibing-Cloud RunPod serverless worker (Hub build).
# Wraps Microsoft's VibeVoice vLLM ASR server behind a RunPod handler.
#
# The ASR checkpoint is expected to come from RunPod's cached-model mount at
# /runpod-volume/huggingface-cache/hub. The image only carries app code plus the
# small tokenizer assets VibeVoice's vLLM path needs.
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
# `vllm serve` directly, so runtime cold start is vLLM boot + cached model load.
RUN pip install --no-cache-dir -e "/app[vllm]"

# Generate VibeVoice's patched tokenizer files without baking the 16GB ASR
# weights. The handler passes this path to vLLM via --tokenizer.
ENV HF_HOME=/models/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1
# Run by file path; the tool is a standalone script, not a package module.
RUN python3 /app/vllm_plugin/tools/generate_tokenizer_files.py --output /app/vibevoice-tokenizer

# Runtime defaults. Gemma off; ASR gets the whole GPU for this spike.
ENV VIBING_GEMMA_ENABLED=0 \
    VIBING_ASR_MODEL=microsoft/VibeVoice-ASR \
    VIBING_ASR_TOKENIZER_PATH=/app/vibevoice-tokenizer \
    VIBING_ASR_GPU_MEM_UTIL=0.88 \
    VIBING_ASR_MAX_MODEL_LEN=256 \
    VIBING_ASR_MAX_NUM_SEQS=1 \
    VIBING_ASR_MAX_NUM_BATCHED_TOKENS=256 \
    VIBING_ASR_MAX_OUTPUT_TOKENS=1024 \
    VIBING_ASR_ENFORCE_EAGER=1 \
    VIBING_ASR_LOCAL_FILES_ONLY=1 \
    VIBING_ASR_ALLOW_RUNTIME_DOWNLOAD=0 \
    MODEL_NAME=microsoft/VibeVoice-ASR \
    VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COPY handler.py /app/handler.py

# RunPod serverless entrypoint (override the base image's vLLM entrypoint).
ENTRYPOINT []
CMD ["python3", "-u", "/app/handler.py"]
