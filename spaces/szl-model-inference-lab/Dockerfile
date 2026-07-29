FROM python:3.11-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    MODEL_DIR_OVERRIDE=/models \
    HOME=/home/user \
    PATH=/home/user/.local/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user
USER user
WORKDIR /home/user/app

COPY --chown=user:user requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --user -r requirements.txt
ENV HF_HUB_OFFLINE=1
COPY --chown=user:user . .

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/live', timeout=4)" || exit 1
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--limit-concurrency", "16", "--timeout-keep-alive", "5", "--no-server-header"]
