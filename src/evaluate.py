#!/usr/bin/env python
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from evaluate import load

TOKENIZER_DIR = "./outputs"
MODEL_DIR     = "./outputs"
TEST_SAMPLES  = 50          # hızlı feedback

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto"
)

test_ds = load_dataset(
    "moremilk/CoT_Temporal_Reasoning_Dataset",
    split=f"train[:{TEST_SAMPLES}]"
)

bleu  = load("bleu")
rouge = load("rouge")

refs, hyps = [], []
for sample in test_ds:
    prompt = sample["question"]
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True
    )
    hyp = tokenizer.decode(out[0], skip_special_tokens=True)
    hyps.append(hyp)
    refs.append(sample["answer"])

bleu_score  = bleu.compute(
    predictions=hyps,
    references=[[r] for r in refs]
)["bleu"]

rouge_score = rouge.compute(
    predictions=hyps,
    references=refs
)["rougeL"]

with open("metrics.json", "w", encoding="utf-8") as f:
    json.dump({"bleu": bleu_score, "rougeL": rouge_score}, f, ensure_ascii=False, indent=2)

print("Evaluation done → metrics.json")
