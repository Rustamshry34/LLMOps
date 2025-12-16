#!/usr/bin/env python3
"""
Push ./outputs (adapter + tokenizer) to HuggingFace Hub
and create a 'prod' tag if BLEU ≥ threshold.
"""
import os
import json
from huggingface_hub import HfApi

MODEL_ID      = os.getenv("MODEL_ID")               # your-hf-username/nizami-1.7b-cot
HF_TOKEN      = os.getenv("HF_TOKEN")
LOCAL_DIR     = "./outputs"
BLEU_THRESHOLD = 0.01   # pipeline.txt ilk iterasyonu için oldukça düşük tutalım

api = HfApi()

# 1. upload folder
api.upload_folder(
    folder_path=LOCAL_DIR,
    repo_id=MODEL_ID,
    repo_type="model",
    token=HF_TOKEN,
    commit_message="Add LoRA adapter & tokenizer"
)

# 2. load metrics.json (evaluate.py sonrası)
try:
    with open("metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    bleu = metrics.get("bleu", 0.0)
except FileNotFoundError:
    bleu = 0.0

# 3. tag prod if threshold passed
if bleu >= BLEU_THRESHOLD:
    api.create_tag(
        repo_id=MODEL_ID,
        tag="prod",
        token=HF_TOKEN,
        exist_ok=True
    )
    print(f"✅ Tagged 'prod' (BLEU={bleu:.3f})")
else:
    print(f"⚠️  BLEU={bleu:.3f} < threshold ({BLEU_THRESHOLD}), prod tag skipped.")

print("Upload complete →", MODEL_ID)
