FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    FOUNDATION_RELEASE_ROOT=/opt/szl/confirmation
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user
WORKDIR /opt/szl/app
COPY --chown=root:root requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes --only-binary=:all: -r requirements.txt
COPY --chown=root:root app.py release.zip ./
RUN python app.py --extract release.zip /opt/szl/confirmation \
    && chmod -R a-w /opt/szl/confirmation /opt/szl/app
COPY --chown=root:root web ./web
RUN chmod -R a-w /opt/szl/app/web
USER 1000:1000
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:7860/readyz', headers={'Host':'szlholdings-szl-foundation-confirmation.hf.space'}), timeout=4)" || exit 1
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--limit-concurrency", "8", "--timeout-keep-alive", "5", "--no-proxy-headers", "--no-server-header"]
