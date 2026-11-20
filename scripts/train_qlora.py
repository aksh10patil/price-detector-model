#!/usr/bin/env python3
"""
CLI script to fine-tune an open-source LLM using QLoRA (4-bit quantization + LoRA).
Corresponds to Week 7 Day 3 & 4 of "The Price is Right" capstone project.
"""

import argparse
import os
from datetime import datetime
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

load_dotenv(override=True)


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning for Price Prediction")
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Base Hugging Face model identifier",
    )
    parser.add_argument(
        "--hf_user",
        type=str,
        default=os.environ.get("HF_USER", "Aksh10"),
        help="Hugging Face username for pushing adapter",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Aksh10/items_prompts_lite",
        help="Hugging Face dataset name with prompt/completion splits",
    )
    parser.add_argument("--lite_mode", action="store_true", default=False, help="Use lite hyperparameters")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Per device train batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum token sequence length")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank r")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--push_to_hub", action="store_true", default=False, help="Push trained adapter to HF")
    parser.add_argument("--no_wandb", action="store_true", default=False, help="Disable Weights & Biases logging")
    return parser.parse_args()


def main():
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(hf_token, add_to_git_credential=True)

    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)
    train_data = dataset["train"]
    val_data = dataset["val"].select(range(min(500, len(dataset["val"]))))

    run_id = f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
    run_name = f"price-{run_id}-lite" if args.lite_mode else f"price-{run_id}"
    hub_model_id = f"{args.hf_user}/{run_name}"

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

    # 4-bit NormalFloat Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    print(f"Loading base model: {args.base_model} in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
    )
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    # LoRA Config
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    if not args.lite_mode:
        target_modules += ["gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # SFT Training Arguments
    training_args = SFTConfig(
        output_dir=run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=args.learning_rate,
        weight_decay=0.001,
        fp16=not use_bf16,
        bf16=use_bf16,
        max_grad_norm=0.3,
        warmup_ratio=0.01,
        lr_scheduler_type="cosine",
        report_to=None if args.no_wandb else "wandb",
        run_name=run_name,
        max_length=args.max_length,
        save_strategy="steps",
        push_to_hub=args.push_to_hub,
        hub_model_id=hub_model_id if args.push_to_hub else None,
        hub_private_repo=True,
        eval_strategy="steps",
        eval_steps=100,
    )

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
        args=training_args,
    )

    print("Beginning fine-tuning...")
    trainer.train()

    if args.push_to_hub:
        print(f"Pushing model adapter to Hugging Face Hub: {hub_model_id}")
        trainer.model.push_to_hub(hub_model_id, private=True)

    trainer.save_model(os.path.join(run_name, "final_adapter"))
    print(f"Fine-tuning complete. Model saved to {run_name}/final_adapter")


if __name__ == "__main__":
    main()
