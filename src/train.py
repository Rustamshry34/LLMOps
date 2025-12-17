#!/usr/bin/env python
import os
import yaml
import torch
import wandb
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from dataset_utils import build_combined_dataset


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    run_name = cfg.get("run_name", "nizami-1.7B-cot")
    wandb.init(project="nizami-llmops", name=run_name)

    # --- model ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        dtype=torch.float16,
        load_in_4bit=False,
        device_map={"": 0}
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=cfg.get("gradient_checkpointing", False)
    )

    # --- data ---
    train_ds = build_combined_dataset(
        tokenizer,
        non_reasoning_pct=0.0,  # pipeline.txt sabiti
        seed=3407
    )

    # --- trainer ---
    training_args = SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=cfg["warmup_steps"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        logging_steps=cfg["logging_steps"],
        fp16=cfg["fp16"],
        report_to=cfg["report_to"],
        run_name=run_name,
        max_seq_length=cfg["max_seq_length"],
        dataset_text_field="text",
        dataloader_num_workers=cfg["dataloader_num_workers"],
        weight_decay=cfg["weight_decay"],
        optim=cfg.get("optim", "adamw_torch")
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds
    )

    trainer.train()
    model.save_pretrained(cfg["output_dir"])  
    tokenizer.save_pretrained(cfg["output_dir"])
    wandb.finish()


if __name__ == "__main__":
    import fire
    fire.Fire(main)
