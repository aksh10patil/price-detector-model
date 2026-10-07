"""Global configuration and hyperparameters for Price Prediction."""

import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
JSONL_DIR = PROJECT_ROOT / "jsonl"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Dataset Settings
DATASET_SOURCE = "McAuley-Lab/Amazon-Reviews-2023"
DEFAULT_HF_USER = os.environ.get("HF_USER", "ed-donner")
DEFAULT_LITE_DATASET = f"{DEFAULT_HF_USER}/items_lite"
DEFAULT_FULL_DATASET = f"{DEFAULT_HF_USER}/items_full"
DEFAULT_PROMPT_LITE = f"{DEFAULT_HF_USER}/items_prompts_lite"
DEFAULT_PROMPT_FULL = f"{DEFAULT_HF_USER}/items_prompts_full"

# Data Filtering Bounds
MIN_PRICE = 0.50
MAX_PRICE = 999.49
MIN_CHARS = 600
MAX_TEXT_EACH = 3000
MAX_TEXT_TOTAL = 4000

REMOVAL_KEYWORDS = [
    "Part Number",
    "Best Sellers Rank",
    "Batteries Included?",
    "Batteries Required?",
    "Item model number",
]

# Prompt Templates
SUMMARIZATION_SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""

QUESTION = "You are an expert product pricer. Estimate the retail price in USD for this product. Respond with only the price as a number."
PREFIX = "Price: $"

# Models
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-3B"
DEFAULT_BATCH_MODEL = "openai/gpt-oss-20b"
TOKEN_CUTOFF = 110

# Evaluation
DEFAULT_EVAL_SIZE = 200
DEFAULT_EVAL_WORKERS = 5
