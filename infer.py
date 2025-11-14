import torch
import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# --- Configuration ---
# Path to the base model on Hugging Face
BASE_MODEL = "bigscience/bloom-560m"

# Absolute path INSIDE THE CONTAINER to your saved adapter
ADAPTER_PATH = "/app/tensorrt_llm/final_bloom_lora_adapter"
DEVICE = "cuda"

def load_model_and_tokenizer():
    """
    Loads the base model, applies the LoRA adapter, and loads the tokenizer.
    """
    print(f"Loading base model: {BASE_MODEL}...")
    # 1. Load the base model in float16
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16
    ).to(DEVICE)

    print(f"Loading LoRA adapter from: {ADAPTER_PATH}...")
    # 2. Load and apply your trained LoRA adapter
    try:
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    except Exception as e:
        print(f"FATAL: Could not load LoRA adapter from {ADAPTER_PATH}.")
        print("Make sure the path is correct inside your container.")
        print(f"Error details: {e}")
        exit()

    # 3. Set the model to evaluation mode (disables dropout, etc.)
    model.eval()
    
    # 4. Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

def run_inference(model, tokenizer, test_file_path, output_file_path, batch_size):
    """
    Loads data from test_file_path, runs inference, and writes the
    next-token probabilities to output_file_path.
    """
    print(f"Starting inference on {test_file_path}...")
    
    # Ensure test file exists
    if not os.path.exists(test_file_path):
        print(f"FATAL: Test file not found at {test_file_path}")
        return

    # Read all lines from the test file
    with open(test_file_path, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]

    # Open the output file
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        
        # Process in batches for efficiency
        for i in range(0, len(test_data), batch_size):
            batch_data = test_data[i:i+batch_size]
            
            # Get the 'input' field from the dataset [cite: 284]
            prompts = [item['input'] for item in batch_data]
            
            # Tokenize the batch
            inputs = tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1024
            ).to(DEVICE)
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits # Shape: [batch_size, seq_len, vocab_size]

            # Get the logits for the *last* token of each item in the batch
            # This is the prediction for the *next* token [cite: 299]
            # We use input_ids.shape[1] to get the sequence length
            seq_len = inputs.input_ids.shape[1]
            last_token_indices = torch.full((batch_size,), seq_len - 1, dtype=torch.long, device=DEVICE)
            
            # Gather the logits for the last token of each sequence
            # (Using gather is more robust for padding)
            last_token_logits = logits[torch.arange(batch_size), last_token_indices, :]
            
            # Convert logits to probabilities using softmax [cite: 299]
            next_token_probs = torch.softmax(last_token_logits, dim=-1)
            
            # Write results for this batch
            probs_lists = next_token_probs.cpu().tolist()
            for item, probs_list in zip(batch_data, probs_lists):
                output_record = {
                    "id": item.get('id', 0), # Use 'id' if available
                    "probabilities": probs_list
                }
                f_out.write(json.dumps(output_record) + '\n')

    print(f"Inference complete. Probabilities written to {output_file_path}")

if __name__ == "__main__":
    # Setup command-line arguments as required by the assignment [cite: 328]
    parser = argparse.ArgumentParser(description="Run inference for Compact LM Challenge")
    
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to the input test .jsonl file")
    parser.add_argument("--output_file", type=str, default="probabilities.jsonl",
                        help="Path to write the output probabilities .jsonl file")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set seed as required by assignment [cite: 328]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load the model
    model, tokenizer = load_model_and_tokenizer()
    
    # Run the inference
    run_inference(model, tokenizer, args.test_file, args.output_file, args.batch_size)