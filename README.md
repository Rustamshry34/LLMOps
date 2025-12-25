# LLMOps Pipeline

[![LLMOps](https://github.com/Rustamshry34/LLMOps/actions/workflows/train-eval-deploy.yaml/badge.svg)](https://github.com/Rustamshry34/LLMOps/actions/workflows/train-eval-deploy.yaml)

End-to-end LLMOps pipeline for fine-tuning, evaluating, and serving a 0.6B-parameter CoT model. The system automates training on Kaggle GPUs, evaluates with multiple metrics (BLEU, ROUGE-L, METEOR, chrF, Perplexity), version-controls model uploads to Hugging Face, and provides a FastAPI + vLLM inference server for production-ready deployments.
