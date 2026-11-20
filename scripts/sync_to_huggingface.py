#!/usr/bin/env python3
"""
Syncs all datasets and fine-tuned LoRA models from ed-donner to user's Hugging Face account (Aksh10).
Eliminates all external third-party dependencies.
"""

import os
import tempfile
from huggingface_hub import HfApi, login, snapshot_download

TOKEN = os.environ.get("HF_TOKEN")
TARGET_USER = os.environ.get("HF_USER", "Aksh10")

print(f"Logging in to Hugging Face as {TARGET_USER}...")
login(token=TOKEN, add_to_git_credential=True)
api = HfApi(token=TOKEN)

TARGETS = [
    # Models (Fine-Tuned QLoRA Adapters)
    {
        "type": "model",
        "src": "ed-donner/price-2025-11-28_18.47.07",
        "dst": f"{TARGET_USER}/price-detector-llama-3.2-3b",
        "desc": "Fine-Tuned Llama 3.2 3B QLoRA Adapter (Full Dataset, $39.85 RMSE)",
    },
    {
        "type": "model",
        "src": "ed-donner/price-2025-11-30_15.10.55-lite",
        "dst": f"{TARGET_USER}/price-detector-llama-3.2-3b-lite",
        "desc": "Fine-Tuned Llama 3.2 3B QLoRA Adapter (Lite Dataset, $65.40 RMSE)",
    },
    # Datasets
    {
        "type": "dataset",
        "src": "ed-donner/items_lite",
        "dst": f"{TARGET_USER}/items_lite",
        "desc": "Pricer Lite Dataset (Items with summaries and prices)",
    },
    {
        "type": "dataset",
        "src": "ed-donner/items_prompts_lite",
        "dst": f"{TARGET_USER}/items_prompts_lite",
        "desc": "Pricer SFT Prompt Dataset (Lite)",
    },
    {
        "type": "dataset",
        "src": "ed-donner/items_prompts_full",
        "dst": f"{TARGET_USER}/items_prompts_full",
        "desc": "Pricer SFT Prompt Dataset (Full)",
    },
    {
        "type": "dataset",
        "src": "ed-donner/items_full",
        "dst": f"{TARGET_USER}/items_full",
        "desc": "Pricer Full Dataset (Items with summaries and prices)",
    },
]


def sync():
    for item in TARGETS:
        t = item["type"]
        src = item["src"]
        dst = item["dst"]
        print(f"\n==========================================")
        print(f"Syncing {t}: {src} -> {dst}")
        print(f"==========================================")

        api.create_repo(repo_id=dst, repo_type=t, exist_ok=True, private=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"Downloading {src}...")
            snapshot_download(repo_id=src, repo_type=t, local_dir=tmpdir)

            print(f"Uploading to {dst}...")
            api.upload_folder(
                folder_path=tmpdir,
                repo_id=dst,
                repo_type=t,
                commit_message=f"Initial import of {item['desc']} from {src}",
            )
            print(f"Successfully synced: https://huggingface.co/{dst}")


if __name__ == "__main__":
    sync()
