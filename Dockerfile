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

# Microsoft's start_server.py lives in the VibeVoice repo; it expects to run from /app.
RUN git clone --depth 1 https://github.com/microsoft/VibeVoice.git /app
WORKDIR /app

# Handler + plugin deps. transformers/pillow are for the (disabled) Gemma path.
RUN pip install --no-cache-dir \
      runpod \
      requests \
      "huggingface_hub[cli]" \
      hf_transfer \
      "transformers>=4.57" \
      pillow \
 && if [ -f /app/vllm_plugin/requirements.txt ]; then \
      pip install --no-cache-dir -r /app/vllm_plugin/requirements.txt ; \
    fi

# Bake the ASR weights into the image's HF cache (no network volume on the Hub).
ENV HF_HOME=/models/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1
RUN huggingface-cli download microsoft/VibeVoice-ASR

# Runtime defaults. Gemma off; ASR gets the whole GPU for this spike.
ENV VIBING_GEMMA_ENABLED=0 \
    VIBING_ASR_GPU_MEM_UTIL=0.90 \
    VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
    PYTORCH_ALLOC_CONF=expandable_segments:True

COPY handler.py /app/handler.py

# RunPod serverless entrypoint (override the base image's vLLM entrypoint).
ENTRYPOINT []
CMD ["python3", "-u", "/app/handler.py"]
