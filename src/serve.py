"""
vLLM + FastAPI inference server
docker run -p 8000:8000 nizami-1.7b
"""
import os
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

MODEL_ID = os.getenv("MODEL_ID", "your-hf-username/nizami-1.7b-cot")

app = FastAPI(title="Nizami-1.7B-CoT")

llm = LLM(
    model=MODEL_ID,
    dtype="float16",
    trust_remote_code=False
)

sampling = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=512
)

class GenerateRequest(BaseModel):
    prompt: str

class GenerateResponse(BaseModel):
    text: str

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    output = llm.generate(req.prompt, sampling)
    return GenerateResponse(text=output[0].outputs[0].text)

@app.get("/health")
def health():
    return {"status": "ok"}
