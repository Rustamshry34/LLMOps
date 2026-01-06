#!/usr/bin/env python
"""
Extended evaluation:
BLEU, ROUGE-L, METEOR, chrF, Perplexity
"""

import json
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from evaluate import load
from tqdm import tqdm
#from peft import PeftModel

TOKENIZER_DIR = "./outputs"
MODEL_DIR     = "./outputs"
TEST_SAMPLES  = 100   # CI için hızlı feedback

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Load model & tokenizer
# --------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)
#model = PeftModel.from_pretrained(base_model, MODEL_DIR).to(device)
model.eval()

# --------------------------------------------------
# Dataset
# --------------------------------------------------
test_ds = load_dataset(
    "musaoc/Quran-reasoning-SFT",
    split=f"train[:{TEST_SAMPLES}]"
)

# --------------------------------------------------
# Metrics
# --------------------------------------------------
bleu   = load("bleu")
rouge  = load("rouge")
meteor = load("meteor")
chrf   = load("chrf")

refs, hyps = [], []
losses = []

# --------------------------------------------------
# Evaluation loop
# --------------------------------------------------
for sample in tqdm(test_ds, desc="Evaluating"):
    prompt = sample["Question"]
    reference = sample["Response"]
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            do_sample=True
        )

    hyp = tokenizer.decode(out[0], skip_special_tokens=True)
    
    def extract_answer(text: str) -> str:
        if "<answer>" in text:
            return text.split("<answer>")[-1].split("</answer>")[0].strip()
        return text.strip()


    hyps.append(extract_answer(hyp))
    refs.append(reference)


    # ---- perplexity ----
    ppl_inputs = tokenizer(
        reference,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)

    with torch.no_grad():
        outputs = model(
            **ppl_inputs,
            labels=ppl_inputs["input_ids"]
        )
        losses.append(outputs.loss.item())

# --------------------------------------------------
# Aggregate metrics
# --------------------------------------------------
bleu_score = bleu.compute(
    predictions=hyps,
    references=[[r] for r in refs]
)["bleu"]

rouge_score = rouge.compute(
    predictions=hyps,
    references=refs
)["rougeL"]

meteor_score = meteor.compute(
    predictions=hyps,
    references=refs
)["meteor"]

try:
    chrf_score = chrf.compute(predictions=hyps, references=refs)["chrf"]
except KeyError:
    chrf_score = 0.0

avg_loss = sum(losses) / len(losses)
perplexity = math.exp(avg_loss)

# --------------------------------------------------
# Save metrics
# --------------------------------------------------
metrics = {
    "bleu": bleu_score,
    "rougeL": rouge_score,
    "meteor": meteor_score,
    "chrf": chrf_score,
    "perplexity": perplexity
}

with open("metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("✅ Evaluation complete → metrics.json")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

