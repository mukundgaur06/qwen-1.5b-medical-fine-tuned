---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
language:
- en
license: apache-2.0
tags:
- base_model:adapter:Qwen/Qwen2.5-1.5B-Instruct
- lora
- qlora
- medical
- sft
- transformers
- trl
---

# Qwen2.5-1.5B Medical Fine-Tuned Assistant

This repository contains a medical domain-adapted version of **Qwen2.5-1.5B-Instruct**, fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) with **QLoRA (4-bit quantization)**. The model is optimized to provide detailed, contextual responses to medical questions based on clinical context.

---

## Model Details

### Model Description

- **Developed by:** Mukund Gaur (`mukundgaur06`)
- **Model Type:** Causal Language Model (Adapter/LoRA)
- **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Language(s):** English
- **License:** Apache 2.0
- **Fine-tuning Tech:** PEFT / QLoRA (4-bit)

### Model Sources

- **Repository:** [mukundgaur06/qwen-1.5b-medical-fine-tuned](https://github.com/mukundgaur06/qwen-1.5b-medical-fine-tuned)

---

## Uses

### Direct Use

This model is intended to assist with natural language processing tasks in the medical domain, such as summarizing patient context, answering clinical queries, and generating context-aware responses.

### Out-of-Scope Use

* **Primary Diagnosis or Medical Advice:** This model is **NOT** a substitute for professional medical advice, diagnosis, or treatment. 
* **Critical Medical Decision-Making:** Never rely on model outputs for emergency medical decisions without human-in-the-loop validation by licensed health professionals.

---

## How to Get Started with the Model

Run the snippet below to load the base model along with your fine-tuned adapter for inference:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. Base Model & Tokenizer
base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. Load Local Adapter Weights
adapter_path = "."  # Direct path to repo folder with adapter_config.json
model = PeftModel.from_pretrained(base_model, adapter_path)

# 3. Inference Example
question = "What are the common symptoms of type 2 diabetes?"
context = "Patient is a 45-year-old male presenting with increased thirst and fatigue."

prompt = f"### Question: {question}\n\n### Context: {context}\n\n### Answer:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response.split("### Answer:")[1].strip())