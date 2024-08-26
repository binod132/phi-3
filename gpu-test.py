import torch
import psutil  # For RAM usage
import os
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

def check_gpu():
    if torch.cuda.is_available():
        print("CUDA is available.")
        print("Number of GPUs:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB")
            print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / (1024**3):.2f} GB")
    else:
        print("CUDA is not available.")

def check_ram():
    ram = psutil.virtual_memory()
    print(f"Total RAM: {ram.total / (1024**3):.2f} GB")
    print(f"Available RAM: {ram.available / (1024**3):.2f} GB")
    print(f"Used RAM: {ram.used / (1024**3):.2f} GB")

def gpu_test():
    if torch.cuda.is_available():
        print("CUDA is available.")
        device = torch.device("cuda")
        tensor = torch.randn(10000, 10000, device=device)
        print("Tensor created on GPU.")
        check_gpu()
        check_ram()
    else:
        print("CUDA is not available.")
def notice():
    print ("import success")

def alpaca_prompt():
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""
    return alpaca_prompt

def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-3-mini-4k-instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    )
    return model, tokenizer
    
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

def main():
    
    # simple gpu use
    # Check GPU
    # Your existing code or training logic goes here
    notice()
    print("Running training script...")
    # For example:''
    model, tokenizer =load_model()
    print("Model loaded...Now preprating dataset")
    # preparing dataset
    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

if __name__ == "__main__":
    main()
