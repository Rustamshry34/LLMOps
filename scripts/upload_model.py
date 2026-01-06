#!/usr/bin/env python3
"""
Push ./outputs (adapter + tokenizer) to HuggingFace Hub
and create a 'prod' tag if BLEU ≥ threshold.
"""
import os
import json
import re
from huggingface_hub import HfApi

MODEL_ID      = os.getenv("MODEL_ID")              
HF_TOKEN      = os.getenv("HF_TOKEN")
LOCAL_DIR     = "./outputs"

# ---- quality thresholds ----

THRESHOLDS = {
    "bleu": 0.03,
    "meteor": 0.15,
    "chrf": 30.0,
    "perplexity": 20.0,
}

api = HfApi(token=HF_TOKEN)

# --------------------------------------------------
# 1. Find latest version tag (vX.Y)
# --------------------------------------------------
def get_next_version(repo_id: str) -> str:
    refs = api.list_repo_refs(repo_id=repo_id, repo_type="model")
    tags = refs.tags
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
except FileNotFoundError:
    print("❌ metrics.json not found → skipping prod promotion")
    exit(0)


# ---- backward compatible parsing ----
gen = metrics.get("generation", metrics)
lm  = metrics.get("language_model", metrics)

bleu = float(gen.get("bleu", 0.0))
meteor = float(gen.get("meteor", 0.0))
chrf = float(gen.get("chrf", 0.0))
ppl = float(lm.get("perplexity", float("inf")))

print(
    f"📊 Metrics → BLEU={bleu:.4f}, "
    f"METEOR={meteor:.4f}, chrF={chrf:.2f}, PPL={ppl:.2f}"
)

# --------------------------------------------------
# 5. Multi-metric quality gate
# --------------------------------------------------
passed = (
    bleu >= THRESHOLDS["bleu"]
    and meteor >= THRESHOLDS["meteor"]
    and chrf >= THRESHOLDS["chrf"]
    and ppl <= THRESHOLDS["perplexity"]
)

if passed:
    api.create_tag(
        repo_id=MODEL_ID,
        tag="prod",
        repo_type="model",
        token=HF_TOKEN,
        exist_ok=True
    )
    print(f"✅ Promoted {next_version} → prod")
else:
    print("⚠️  Quality gate failed → prod not updated")

print(f"🚀 Upload complete → {MODEL_ID}@{next_version}")
