#!/usr/bin/env python3
"""
CLI script to evaluate a fine-tuned model or adapter on the test set.
Corresponds to Week 7 Day 5 evaluation.
"""

import argparse
import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import PeftModel
from util import evaluate

load_dotenv(override=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Price Prediction Model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Base model identifier",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="Aksh10/price-detector-llama-3.2-3b",
        help="Path or HuggingFace ID for trained LoRA adapter",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Aksh10/items_prompts_lite",
        help="Dataset with test split",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=200,
        help="Number of test samples to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(hf_token, add_to_git_credential=True)

    print(f"Loading dataset {args.dataset_name} test split...")
    dataset = load_dataset(args.dataset_name)
    test_data = dataset["test"]

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    print(f"Loading tokenizer & base model {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
    )
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    if args.adapter_path:
        print(f"Loading LoRA adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
    else:
        print("No adapter specified. Evaluating base model...")
        model = base_model

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    def model_predict(item):
        prompt = item["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device if "cuda" in device else "cpu")
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=8)
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, prompt_len:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"Starting evaluation on {args.test_size} items...")
    evaluate(model_predict, test_data, size=args.test_size)


if __name__ == "__main__":
    main()
