# -------------------------------------------------
# Builder
# -------------------------------------------------
FROM python:3.10-slim as builder

ENV PYTHONUNBUFFERED=1
WORKDIR /build

COPY requirements-serving.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-serving.txt

# -------------------------------------------------
# Runtime
# -------------------------------------------------
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy installed packages
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application
COPY src/serve.py /app/serve.py

# Environment
ENV MODEL_ID=Rustamshry/Qwen3-CoT

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]

