#!/usr/bin/env python3
"""
Push ./outputs (adapter + tokenizer) to HuggingFace Hub
and create a 'prod' tag if BLEU ≥ threshold.
"""
import os
import json
import re
from huggingface_hub import HfApi

MODEL_ID      = os.getenv("MODEL_ID")               # your-hf-username/nizami-1.7b-cot
HF_TOKEN      = os.getenv("HF_TOKEN")
LOCAL_DIR     = "./outputs"
BLEU_THRESHOLD = 0.01   # pipeline.txt ilk iterasyonu için oldukça düşük tutalım

api = HfApi(token=HF_TOKEN)

# --------------------------------------------------
# 1. Find latest version tag (vX.Y)
# --------------------------------------------------
def get_next_version(repo_id: str) -> str:
    tags = api.list_repo_tags(repo_id=repo_id, repo_type="model")
    versions = []

    for tag in tags:
        match = re.match(r"v(\d+)\.(\d+)", tag.name)
        if match:
            major, minor = map(int, match.groups())
            versions.append((major, minor))

    if not versions:
        return "v1.0"

    major, minor = max(versions)
    return f"v{major}.{minor + 1}"


next_version = get_next_version(MODEL_ID)
print(f"📦 Next model version → {next_version}")



# 1. upload folder
api.upload_folder(
    folder_path=LOCAL_DIR,
    repo_id=MODEL_ID,
    repo_type="model",
    token=HF_TOKEN,
    commit_message=f"Add LoRA adapter & tokenizer({next_version})"
)

# --------------------------------------------------
# 3. Create version tag
# --------------------------------------------------
api.create_tag(
    repo_id=MODEL_ID,
    tag=next_version,
    repo_type="model",
    token=HF_TOKEN,
    exist_ok=False
)

print(f"🏷️  Created tag {next_version}")


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
