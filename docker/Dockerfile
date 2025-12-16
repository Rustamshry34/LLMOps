# Multi-stage: build = pip install, runtime = vllm
FROM nvcr.io/nvidia/pytorch:23.10-py3 as builder

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# --- Runtime ---
FROM nvcr.io/nvidia/pytorch:23.10-py3
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY --from=builder /opt/conda/lib/python3.10/site-packages /opt/conda/lib/python3.10/site-packages
COPY --from=builder /opt/conda/bin /opt/conda/bin

COPY src/serve.py /app/
ENV MODEL_ID=${MODEL_ID:-your-hf-username/nizami-1.7b-cot}
EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
