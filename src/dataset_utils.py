
from datasets import load_dataset, Dataset
import pandas as pd


def generate_conversation(examples):
    problems  = examples["question"]
    metadata  = examples["metadata"]
    answers   = examples["answer"]
    conversations = []
    for p, m, a in zip(problems, metadata, answers):
        resp = f"<think>{m.get('reasoning', '')}</think>\n\n<answer>{a}</answer>"
        conversations.append([
            {"role": "user",      "content": p},
            {"role": "assistant", "content": resp}
        ])
    return {"conversations": conversations}


def build_combined_dataset(tokenizer, non_reasoning_pct: float = 0.0, seed=3407):
    from unsloth.chat_templates import standardize_sharegpt
    # ---------- reasoning ----------
    reasoning_ds = load_dataset(
        "moremilk/CoT_Temporal_Reasoning_Dataset",
        split="train"
    )
    reasoning_ds = reasoning_ds.map(
        generate_conversation,
        batched=True,
        remove_columns=reasoning_ds.column_names
    )
    reasoning_text = [
        tokenizer.apply_chat_template(conv, tokenize=False)
        for conv in reasoning_ds["conversations"]
    ]

    # ---------- non-reasoning ----------
    non_ds = load_dataset("mlabonne/FineTome-100k", split="train")
    non_ds = standardize_sharegpt(non_ds)
    non_text = tokenizer.apply_chat_template(
        non_ds["conversations"],
        tokenize=False
    )
    n_non = int(len(non_text) * non_reasoning_pct)
    if n_non:
        non_text = non_text.sample(n=n_non, random_state=seed)

    # ---------- concat ----------
    data = pd.Series(reasoning_text + non_text.tolist() if hasattr(non_text, "tolist") else reasoning_text)
    data.name = "text"
    return Dataset.from_pandas(pd.DataFrame(data)).shuffle(seed=seed)

