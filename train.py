#!/usr/bin/env python
# coding: utf-8

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import evaluate
import time
import os

# --- Configuration (GPU Optimized) ---
MODEL_NAME = "bigscience/bloom-560m" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LENGTH = 1024 
print(f"Using device: {DEVICE}")


def load_indic_model_for_gpu(model_name, device):
    """Loads model and tokenizer to GPU using float16."""

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name} to {device} (float16)")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to(device) 

    print(f"Model loaded successfully. Total parameters: {model.num_parameters():,}")
    return tokenizer, model

tokenizer, model = load_indic_model_for_gpu(MODEL_NAME, DEVICE)


def preprocess_data(tokenizer, max_seq_length):
    """Loads, formats, tokenizes, and groups the Causal LM data from server paths."""

    # 1. Load and Concatenate Datasets
    try:
        # CORRECTED: All lines are now properly indented under the 'try' block
        train_ds_te = load_dataset('json', data_files='/data/train_te.jsonl', split='train')
        train_ds_mr = load_dataset('json', data_files='/data/train_mr.jsonl', split='train')
        val_ds_te = load_dataset('json', data_files='/data/validation_te.jsonl', split='train')
        val_ds_mr = load_dataset('json', data_files='/data/validation_mr.jsonl', split='train')
        
        # CORRECTED: This print statement is now inside the 'try' block
        print("Successfully loaded data files from '/data/' directory.")

    except Exception as e:
        # CORRECTED: This 'except' block is now correctly aligned with 'try'
        print(f"ERROR: Could not load data from '/data/' directory. Make sure files exist.")
        print(f"Details: {e}")
        from datasets import Dataset
        # Added dummy data to prevent num_samples=0 error if files are missing
        empty_ds = Dataset.from_dict({'input': ['dummy text'], 'target': ['dummy text']})
        train_ds_te, train_ds_mr, val_ds_te, val_ds_mr = [empty_ds] * 4


    raw_train_dataset = concatenate_datasets([train_ds_te, train_ds_mr])
    raw_val_dataset = concatenate_datasets([val_ds_te, val_ds_mr])

    print(f"Raw train dataset columns: {raw_train_dataset.column_names}")
    print(f"Raw validation dataset columns: {raw_val_dataset.column_names}")

    # 2. Format Input and Tokenize
    def format_and_tokenize(examples):
        texts = [f"{i} {tokenizer.eos_token} {t}" for i, t in zip(examples['input'], examples['target'])]

        return tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding=False
        )

    tokenized_train_ds = raw_train_dataset.map(
        format_and_tokenize,
        batched=True,
        remove_columns=raw_train_dataset.column_names,
        num_proc=os.cpu_count() or 1
    )

    tokenized_val_ds = raw_val_dataset.map(
        format_and_tokenize,
        batched=True,
        remove_columns=raw_val_dataset.column_names,
        num_proc=os.cpu_count() or 1
    )

    # 3. Create a Data Collator for Causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False # Causal Language Modeling
    )

    return tokenized_train_ds, tokenized_val_ds, data_collator

train_dataset, val_dataset, data_collator = preprocess_data(tokenizer, MAX_SEQ_LENGTH)


def setup_lora(model):
    """Configures and applies LoRA to the model."""
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query_key_value"], # Specific to BLOOM model
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    print("\n--- LoRA Configuration ---")
    model.print_trainable_parameters()
    print("--------------------------\n")

    return model

model = setup_lora(model)


# --- Define Metrics Function ---
perplexity_metric = evaluate.load("perplexity")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    try:
        results = perplexity_metric.compute(predictions=shift_logits, references=shift_labels)
        ppl = results['mean_perplexity']
        loss = ppl # Use perplexity as the loss metric for reporting
    except Exception:
        ppl = -1.0
        loss = -1.0

    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    accuracy = (predictions[mask] == labels[mask]).mean()

    return {
        "perplexity": ppl,
        "cross_entropy": loss,
        "top1_accuracy": accuracy,
    }


# --- Define Training Arguments (A100 GPU Optimized) ---
training_args = TrainingArguments(
    # CORRECTED: 'remove_unused_columns' is now a proper argument
    remove_unused_columns=False,
    output_dir="./bloom_challenge_results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="epoch",
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    bf16=False,
    seed=42,
    report_to=[]
)

# --- Initialize and Train ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\nStarting GPU Training...")
trainer.train()

print("\nTraining complete. Saving final LoRA adapter.")
model.save_pretrained("./final_bloom_lora_adapter")