# CS6320E: Compact Language Model Challenge

This repository contains the complete implementation for the CS6320E Compact Language Model Challenge. [cite_start]The objective is to build a compact, efficient, and sample-efficient causal language model (LLM) for next-token prediction in the low-resource languages **Telugu** and **Marathi**. [cite: 3]

This solution uses a pre-trained **`bigscience/bloom-560m`** model, which is fine-tuned using **LoRA (Low-Rank Adaptation)** for parameter-efficient training. The entire workflow is designed for reproducible execution on an NVIDIA A100 GPU using a Docker container.

---

## 📂 Project Structure

This repository follows the submission structure required by the assignment:

* [cite_start]`/model/final_bloom_lora_adapter/`: Contains the final trained LoRA adapter weights. [cite: 57]
* `train.py`: Script used to train the model and save the adapter.
* [cite_start]`infer.py`: Inference script that loads the adapter and outputs next-token probabilities for a given test file. 
* [cite_start]`measure_performance.py`: Script to load the adapter and report all required efficiency metrics (parameters, latency, memory, etc.). [cite: 61]
* [cite_start]`requirements.txt`: A list of all necessary Python dependencies. [cite: 65]
* [cite_start]`README.md`: This file, containing reproduction instructions. [cite: 63]
* [cite_start]`report.pdf`: (Pending) The 4-6 page technical report. [cite: 64]

---

## ⚙️ Hardware & Environment

* **Hardware:** NVIDIA A100 80GB GPU
* **Environment:** NVIDIA Docker
* **Base Container Image:** `nvcr.io/nvidia/tensorrt-llm/release:latest`

---

## 🚀 How to Reproduce Results

Follow these steps on the GPU server to set up the environment and run the evaluation scripts.

### 1. Setup (on GPU Server)

First, clone this repository and place the model adapter in the correct directory.

```bash
# 1. Clone this repository
git clone [YOUR_GITHUB_REPO_URL]
cd [YOUR_REPO_NAME]

# 2. Create the 'model' directory as per the submission structure
mkdir model

# 3. Move your trained adapter (from the previous run) into this directory
# (This path assumes you trained the model in the tensorrt_llm directory as before)
mv ~/tensorrt_llm/final_bloom_lora_adapter ~/YOUR_REPO_NAME/model/