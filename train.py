# Prepare your training script
import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# Model and tokenizer setup
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-3-mini-4k-instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Prepare dataset and trainer
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        max_steps=60,
        output_dir="/tmp/model_output",
    ),
)

# Train and save the model
trainer.train()
model.save_pretrained("/tmp/model_output")
tokenizer.save_pretrained("/tmp/model_output")
