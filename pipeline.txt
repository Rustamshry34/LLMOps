
# Import libraries
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
import torch
from huggingface_hub import notebook_login,login
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt


notebook_login()

# Model loading 
model, tokenizer = FastLanguageModel.from_pretrained(
  model_name="unsloth/Qwen3-1.7B",
  max_seq_length=5000,
  dtype=torch.float16,
  load_in_4bit=False,
  device_map={"": 0}
)

#PEFT
model = FastLanguageModel.get_peft_model(
 model,
 r=16,
 lora_alpha=32,
 target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
"gate_proj", "up_proj", "down_proj",],
 lora_dropout=0.0,
 bias="none",
 use_gradient_checkpointing = False,
)


# loading the data
reasoning_dataset = load_dataset("moremilk/CoT_Rare-Diseases_And_Health-Conditions", split="train")
non_reasoning_dataset = load_dataset("mlabonne/FineTome-100k", split="train")


#Formatting
def generate_conversation(examples):
 problems = examples["question"]
 metadata = examples["metadata"]
 answers = examples["answer"]
 conversations = []
for problem, meta, answer in zip(problems, metadata, answers):
 assistant_response = (
f"<think>{meta.get('reasoning', '')}</think>\n\n"
f"<answer>{answer}</answer>"
)
 conversations.append([
{"role": "user", "content": problem},
{"role": "assistant", "content": assistant_response},
])
return {"conversations": conversations}


reasoning_dataset = reasoning_dataset.map(generate_conversation, batched=True)
reasoning_conversation = [
 tokenizer.apply_chat_template(conv, tokenize=False)
for conv in reasoning_dataset["conversations"]
]


dataset = standardize_sharegpt(non_reasoning_dataset)
non_reasoning_conversations = tokenizer.apply_chat_template(
 dataset["conversations"],
 tokenize = False,
)


import pandas as pd
non_reasoning = pd.Series(non_reasoning_conversations)
#n = int(len(non_reasoning) * chat_percentage)
non_reasoning_subset = non_reasoning.sample(
0,
 random_state=2407,
 ignore_index=True
)


data = pd.concat([
 pd.Series(reasoning_conversation)
])
data.name = "text"
from datasets import Dataset
combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
combined_dataset = combined_dataset.shuffle(seed = 3407)


#Training config: 
training_args = SFTConfig(
 output_dir="./Nizami-1.7B",
 per_device_train_batch_size=4,
 gradient_accumulation_steps=4,
 optim="adamw_torch",
 learning_rate=3e-5,
 logging_steps=10,
 fp16=True,
 report_to="none",
 gradient_checkpointing = False,
 dataloader_num_workers=16,
 num_train_epochs=2,
 weight_decay=0.01,
 warmup_steps=61,
 lr_scheduler_type = "cosine"
)


trainer = SFTTrainer(
 model=model,
 tokenizer=tokenizer,
 args=training_args,
 train_dataset=combined_dataset,
 dataset_text_field="text",
 max_seq_length=5000
)

trainer.train()

#Inference


question="""
How is the D-dimer assay utilized in the context of suspected Venous Thromboembolism (VTE), and what are the clinical implications of a negative result, particularly concerning the Wells' score in clinical decision-making algorithms?
"""
messages = [
{"role" : "user", "content" : question}
]
text = tokenizer.apply_chat_template(
 messages,
 tokenize = False,
 add_generation_prompt = True, # Must add for generation
 enable_thinking = True,
)
from transformers import TextStreamer
_ = model.generate(
**tokenizer(text, return_tensors = "pt").to("cuda"),
 max_new_tokens = 2500,
 temperature = 0.6,
 top_p = 0.95,
 top_k = 20,
 streamer = TextStreamer(tokenizer, skip_prompt = True),
)








