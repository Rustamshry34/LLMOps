"""
Transformers + FastAPI inference server (CPU-only)

docker run -p 8000:8000 nizami-1.7b
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = os.getenv("MODEL_ID", "Rustamshry/Qwen3-CoT")

app = FastAPI(title="Nizami-1.7B-CoT (CPU Inference)")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    device_map={"": "cpu"},
    trust_remote_code=True,
    dtype=torch.float32
)
model = PeftModel.from_pretrained(base_model, MODEL_ID).to("cpu")
model.eval()


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None


class GenerateResponse(BaseModel):
    text: str


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    messages = [
        {"role": "user", "content": req.prompt}
    ]

    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        inputs = tokenizer(
            prompt_text,
            return_tensors="pt"
        ).to("cpu")

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens or 1024,
                temperature=req.temperature or 0.6,
                top_p=req.top_p or 0.95,
                top_k=req.top_k or 20,
            )

        generated_text = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        return GenerateResponse(text=generated_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "device": "cpu"}

