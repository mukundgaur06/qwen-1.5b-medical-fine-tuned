import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. Load Base Model and Tokenizer
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "."  # Direct path to directory containing adapter_config.json

print("Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading fine-tuned PEFT adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

# 2. Inference Function
def generate_medical_response(question, context=""):
    prompt = f"### Question: {question}\n\n### Context: {context}\n\n### Answer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
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
    return response.split("### Answer:")[1].strip()

# 3. Gradio Interface
interface = gr.Interface(
    fn=generate_medical_response,
    inputs=[
        gr.Textbox(lines=3, placeholder="Enter medical question...", label="Question"),
        gr.Textbox(lines=3, placeholder="Enter patient context (optional)...", label="Context")
    ],
    outputs=gr.Textbox(lines=6, label="Fine-Tuned Response"),
    title="Qwen Medical AI Assistant"
)

if __name__ == "__main__":
    interface.launch(share=True)