FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

RUN python -m pip install --disable-pip-version-check --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch==2.9.1" \
    && python -m pip install --disable-pip-version-check --no-cache-dir \
      "huggingface-hub==1.26.0" \
      "kernels==0.16.0"

WORKDIR /runtime
COPY tools/publish_szl_kernels.py tools/verify_szl_kernel_runtime.py ./

USER 65532:65532
ENTRYPOINT ["python", "/runtime/verify_szl_kernel_runtime.py"]
