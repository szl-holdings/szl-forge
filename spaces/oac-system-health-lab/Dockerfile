# SPDX-License-Identifier: Apache-2.0
# Digest matches the existing canonical A11oy Python runtime image.
FROM python:3.14-slim@sha256:caaf356f40667c496d405780745b9ac25771c189a51dfcc42430d531ea09f8a2
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY app.py verify_release.py index.html release.json LICENSE README.md ./
COPY artifacts/ ./artifacts/
USER 1000:1000
EXPOSE 7860
CMD ["python", "-I", "-B", "app.py", "--host", "0.0.0.0", "--port", "7860"]
