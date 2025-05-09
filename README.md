# HF LLaMA Double Quantization

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-blue.svg)](https://huggingface.co/premkumarkora/kora-2-2b-it)

## Overview

**HF LLaMA Double Quantization** provides an end-to-end example of loading, quantizing, and deploying LLaMA-based models using Hugging Face Transformers and BitsAndBytes.  
This repository demonstrates how to reduce memory footprint and accelerate inference by applying 4-bit NormalFloat (NF4) quantization with double quantization, leveraging bfloat16 for compute.

## Key Features

- **Model Support**: Works with popular LLaMA-family models (e.g., LLaMA-2 7B/13B/70B).  
- **4-bit NF4 Quantization**: Applies NormalFloat-4 quant with double quantization for optimal accuracy-memory tradeoff.  
- **Memory Efficiency**: Achieves >75% reduction in GPU RAM usage.  
- **BFloat16 Compute**: Uses bfloat16 during runtime for stable and efficient matrix operations.  
- **Streaming Inference**: Integrates `TextStreamer` for real-time token streaming.  
- **Device Mapping**: Automatic device placement across GPUs/CPUs via `device_map="auto"`.  

## Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Installation](#installation)  
3. [Quantization Configuration](#quantization-configuration)  
4. [Usage](#usage)  
5. [Results & Metrics](#results--metrics)  
6. [Troubleshooting](#troubleshooting)  
7. [License](#license)  
8. [Citation](#citation)  
9. [Contact](#contact)  

## Prerequisites

- Python ≥ 3.9  
- CUDA-enabled GPU (compute capability ≥ 7.0 recommended)  
- Approximately 16 GB of GPU memory for medium-sized models  

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/premkumarkora/HF_LLaMA_Double_Quantization.git
cd HF_LLaMA_Double_Quantization
pip install -r requirements.txt
```

**requirements.txt** should include:
```text
torch>=2.0
transformers>=4.30
bitsandbytes>=0.39
sentencepiece
accelerate
huggingface_hub
```

## Quantization Configuration

Below is the recommended BitsAndBytes quantization configuration:

```python
from bitsandbytes import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_quant_type="nf4"
)
```

- **load_in_4bit**: Enables 4-bit weight loading  
- **bnb_4bit_use_double_quant**: Applies two-stage quantization for scale factors  
- **bnb_4bit_compute_dtype**: Uses bfloat16 for runtime computations  
- **bnb_4bit_quant_type**: Sets the quantization scheme to NF4  

## Usage

1. **Load and Quantize the Model**  
   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM
   from bitsandbytes import BitsAndBytesConfig

   tokenizer = AutoTokenizer.from_pretrained("facebook/llama-7b")
   model = AutoModelForCausalLM.from_pretrained(
       "facebook/llama-7b",
       quantization_config=quant_config,
       device_map="auto"
   )
   ```

2. **Define a Generation Function**  
   ```python
   from transformers import TextStreamer
   import torch, gc

   def generate(model, tokenizer, prompt: str, max_new_tokens: int = 50):
       inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")
       streamer = TextStreamer(tokenizer)
       outputs = model.generate(
           **inputs, max_new_tokens=max_new_tokens, streamer=streamer
       )
       # Cleanup
       del inputs, streamer
       gc.collect()
       torch.cuda.empty_cache()
       return outputs
   ```

3. **Run Inference**  
   ```python
   prompt = "Once upon a time"
   generate(model, tokenizer, prompt)
   ```

## Results & Metrics

| Model Variant      | FP16 Memory (MB) | 4-bit NF4 Memory (MB) | Reduction (%) |
|--------------------|------------------|-----------------------|---------------|
| llama-7b           | 13,000           | ~3,200                | 75.4%         |
| llama-13b          | 26,000           | ~6,400                | 75.4%         |

Memory measured using:
```python
total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
memory_mb   = total_bytes / 1e6
print(f"Memory footprint: {memory_mb:,.1f} MB")
```

## Troubleshooting

- **Tokenizers Parallelism Warning**  
  Set at the top of your script:
  ```python
  import os
  os.environ["TOKENIZERS_PARALLELISM"] = "false"
  ```
- **CUDA OOM**  
  - Reduce `max_new_tokens` or batch sizes  
  - Offload layers to CPU via `device_map` configurations  

## License

This project is licensed under the [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0).

## Citation

```bibtex
@misc{hf_llama_double_quantization,
  title        = {HF LLaMA Double Quantization: Efficient 4-bit NF4 Inference},
  author       = {premkumarkora},
  year         = {2025},
  howpublished = {\url{https://github.com/premkumarkora/HF_LLaMA_Double_Quantization}}
}
```

## Contact

For questions or contributions, please open an issue or pull request on GitHub.
