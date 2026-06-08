# Vibing-Cloud RunPod serverless worker.
# Wraps Microsoft's VibeVoice vLLM ASR server + Gemma-4 polish behind a RunPod handler.
# Base image is Microsoft's documented target for the vLLM ASR plugin.
FROM vllm/vllm-openai:v0.14.1

# Microsoft's start_server.py lives in the VibeVoice repo; it expects to run from /app.
# Pin a commit in production once the spike validates; main is fine for the spike.
RUN git clone --depth 1 https://github.com/microsoft/VibeVoice.git /app
WORKDIR /app

# Plugin + handler deps. transformers/pillow are for the in-process Gemma path.
RUN pip install --no-cache-dir \
      runpod \
      requests \
      "transformers>=4.57" \
      pillow \
 && if [ -f /app/vllm_plugin/requirements.txt ]; then \
      pip install --no-cache-dir -r /app/vllm_plugin/requirements.txt ; \
    fi

# Keep HF weights on the mounted network volume so a 16GB+ ASR checkpoint is
# downloaded ONCE and reused across cold starts (set the volume mount to /runpod-volume).
ENV HF_HOME=/runpod-volume/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
    PYTORCH_ALLOC_CONF=expandable_segments:True

COPY handler.py /app/handler.py

# RunPod serverless entrypoint. --entrypoint is overridden vs the base image.
ENTRYPOINT []
CMD ["python3", "-u", "/app/handler.py"]
