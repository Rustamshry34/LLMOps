def generate_conversation(examples):
    problems  = examples["Question"]
    metadata  = examples["Complex_CoT"]
    answers   = examples["Response"]
    conversations = []
    for p, m, a in zip(problems, metadata, answers):
        resp = (
            f"<think>{m}</think>\n\n"
            f"<answer>{a}</answer>" 
        )
        conversations.append([
            {"role": "user",      "content": p},
            {"role": "assistant", "content": resp}
        ])
    return {"conversations": conversations}

def build_combined_dataset(tokenizer, non_reasoning_pct: float = 0.0, seed=3407):
    from datasets import load_dataset, Dataset
    import pandas as pd
    # ---------- reasoning ----------
    reasoning_ds = load_dataset(
        "musaoc/Quran-reasoning-SFT",
        split="train"
    )
    reasoning_ds = reasoning_ds.map(
        generate_conversation,
        batched=True
    )
    reasoning_conversation = [
        tokenizer.apply_chat_template(conv, tokenize=False)
        for conv in reasoning_ds["conversations"]
    ]

    # ---------- concat ----------
    data = pd.concat([pd.Series(reasoning_conversation),])
    data.name = "text"
    return Dataset.from_pandas(pd.DataFrame(data)).shuffle(seed=seed)

