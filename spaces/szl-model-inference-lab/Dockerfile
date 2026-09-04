FROM python:3.11-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user
USER user
WORKDIR /home/user/app

COPY --chown=user:user requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --user -r requirements.txt

FROM runtime-base AS artifact-builder
COPY --chown=user:user download_artifacts.py ./download_artifacts.py
# Build only: fetch four public files from one immutable model revision, then
# fail the image build unless their declared sizes and SHA-256 digests match.
RUN python download_artifacts.py --fetch --output-dir /home/user/model-artifacts

FROM runtime-base AS runtime
ENV HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_XET=1
COPY --from=artifact-builder --chown=root:root /home/user/model-artifacts/ /opt/szl/model-artifacts/
COPY --chown=user:user . .

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/live', timeout=4)" || exit 1
CMD ["python", "-m", "uvicorn", "app_governed:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--limit-concurrency", "16", "--timeout-keep-alive", "5", "--no-server-header"]
