"""QLoRA fine-tuning pipeline for causal language models (Llama 3.2 3B)."""

import os
import argparse
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
from pricer.config import DEFAULT_BASE_MODEL, DEFAULT_PROMPT_LITE

load_dotenv(override=True)


def train_qlora(
    base_model: str = DEFAULT_BASE_MODEL,
    dataset_name: str = DEFAULT_PROMPT_LITE,
    hf_user: str = "my-user",
    lite_mode: bool = True,
    epochs: int = 1,
    batch_size: int = 16,
    lr: float = 1e-3,
    max_length: int = 128,
    push_to_hub: bool = False,
    use_wandb: bool = False,
):
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(hf_token, add_to_git_credential=True)

    print(f"Loading SFT prompt dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name)
    train_data = dataset["train"]
    val_data = dataset["val"].select(range(min(500, len(dataset["val"]))))

    run_id = f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
    run_name = f"price-{run_id}-lite" if lite_mode else f"price-{run_id}"
    hub_model_id = f"{hf_user}/{run_name}"

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    base.generation_config.pad_token_id = tokenizer.pad_token_id

    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    if not lite_mode:
        target_modules += ["gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=32 if lite_mode else 256,
        lora_alpha=64 if lite_mode else 512,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    training_args = SFTConfig(
        output_dir=run_name,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit" if torch.cuda.is_available() else "adamw_torch",
        save_steps=100,
        logging_steps=10,
        learning_rate=lr,
        weight_decay=0.001,
        fp16=not use_bf16 and torch.cuda.is_available(),
        bf16=use_bf16,
        max_grad_norm=0.3,
        warmup_ratio=0.01,
        lr_scheduler_type="cosine",
        report_to="wandb" if use_wandb else None,
        run_name=run_name,
        max_length=max_length,
        save_strategy="steps",
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id if push_to_hub else None,
        hub_private_repo=True,
        eval_strategy="steps",
        eval_steps=100,
    )

    trainer = SFTTrainer(
        model=base,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
        args=training_args,
    )

    print("Beginning QLoRA training...")
    trainer.train()

    if push_to_hub:
        trainer.model.push_to_hub(hub_model_id, private=True)

    trainer.save_model(os.path.join(run_name, "final_adapter"))
    print(f"Training completed successfully. Checkpoint saved to {run_name}/final_adapter")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset", type=str, default=DEFAULT_PROMPT_LITE)
    parser.add_argument("--lite_mode", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    args = parser.parse_args()
    train_qlora(
        base_model=args.base_model,
        dataset_name=args.dataset,
        lite_mode=args.lite_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        push_to_hub=args.push_to_hub,
    )
