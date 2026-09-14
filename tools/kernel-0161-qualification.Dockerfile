# Independent qualification image. The canonical 0.16.0 publisher is unchanged.
FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7
RUN python -m pip install --disable-pip-version-check --no-cache-dir \
      --report /torch-install-report.json \
      --index-url https://download.pytorch.org/whl/cpu "torch==2.9.1" \
    && python -m pip install --disable-pip-version-check --no-cache-dir \
      --report /dependency-install-report.json \
      "huggingface-hub==1.26.0" "numpy==2.3.5" "kernels==0.16.1"
WORKDIR /runtime
COPY tools/qualify_kernel_0161.py tools/publish_szl_kernels.py ./
# Deliberately exclude .git, credentials, unrelated source and business data.
COPY source-candidate/build/torch-universal/szl_kernels/_kernel_api.py \
     source-candidate/build/torch-universal/szl_kernels/_chain.py \
     source-candidate/build/torch-universal/szl_kernels/_ops.py \
     source-candidate/build/torch-universal/szl_kernels/retrieval.py /admitted-source/
ENV PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=12 HF_HUB_ETAG_TIMEOUT=12 HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/cache OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
USER 65532:65532
ENTRYPOINT ["python", "/runtime/qualify_kernel_0161.py"]
