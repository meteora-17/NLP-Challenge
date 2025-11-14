import torch
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# --- Configuration ---
# Path to the base model on Hugging Face
BASE_MODEL = "bigscience/bloom-560m"

# Absolute path INSIDE THE CONTAINER to your saved adapter
# We know it's here from your previous 'ls' command
ADAPTER_PATH = "/app/tensorrt_llm/final_bloom_lora_adapter"

DEVICE = "cuda"

# --- 1. Load Model and Tokenizer ---

print(f"Loading base model: {BASE_MODEL}...")
# Load the base model in float16 for efficiency
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16
).to(DEVICE)

print(f"Loading LoRA adapter from: {ADAPTER_PATH}...")
# Load and apply your trained LoRA adapter on top
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

# Set model to evaluation mode (disables dropout, etc.)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
print("Model and tokenizer loaded successfully.")


# --- 2. Calculate Parameter Counts ---
# [cite: 37]
print("\n--- 2. Parameter Counts ---")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Parameters (Base + Adapter): {total_params:,}")
print(f"Trainable Parameters (Adapter only): {trainable_params:,}")


# --- 3. Calculate Latency & Throughput ---
# [cite: 38, 39]
print("\n--- 3. Latency & Throughput Test ---")
prompt = "క్రవరి 23, 2019 174 ఎన్నో భారీ అంచనాలతో" # Sample Telugu prompt
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

latencies = []
# Warmup run: Run once to load everything into CUDA memory
print("Running warmup iteration...")
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=2, pad_token_id=tokenizer.eos_token_id)

print(f"Running latency test (50 iterations)...")
for _ in range(50):
    torch.cuda.synchronize() # Wait for GPU to be ready
    start_time = time.time()
    
    with torch.no_grad():
        # Generate one new token to measure per-token latency
        _ = model.generate(**inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
    
    torch.cuda.synchronize() # Wait for GPU to finish
    end_time = time.time()
    latencies.append(end_time - start_time)

# Calculate metrics in milliseconds
median_latency_ms = np.median(latencies) * 1000
p95_latency_ms = np.percentile(latencies, 95) * 1000
throughput_tokens_sec = 1 / np.median(latencies)

print(f"Median Latency (p50): {median_latency_ms:.2f} ms per token")
print(f"p95 Latency:          {p95_latency_ms:.2f} ms per token")
print(f"Throughput:           {throughput_tokens_sec:.2f} tokens/sec")


# --- 4. Calculate Peak GPU Memory ---
# [cite: 42]
print("\n--- 4. Peak GPU Memory ---")
# torch.cuda.max_memory_allocated() tracks the high-water mark of memory usage
mem_mib = torch.cuda.max_memory_allocated(DEVICE) / (1024**2)
print(f"Peak GPU Memory Allocated: {mem_mib:.2f} MiB")


# --- 5. FLOPS Estimation ---
# [cite: 40]
print("\n--- 5. FLOPS/MACs ---")
print("FLOPS/MACs: This must be reported in your PDF.")
print("For your report, state your 'method used for estimation'.")
print("Method 1: Use a library like 'thop' (run 'pip install thop').")
print("Method 2: Cite the FLOPS from the 'bloom-560m' model card or paper.")
print("Method 3: Calculate TFLOPs = (Params * 2 * Seq_Length) * Throughput (Simplified).")