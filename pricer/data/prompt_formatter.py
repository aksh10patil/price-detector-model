from typing import Tuple
from tqdm import tqdm
from pricer.config import DEFAULT_BASE_MODEL, TOKEN_CUTOFF
from pricer.data.schema import Item


class PromptFormatter:
    """Handles token counting, cutoff analysis, and prompt/completion preparation."""

    def __init__(self, base_model: str = DEFAULT_BASE_MODEL, cutoff: int = TOKEN_CUTOFF):
        from transformers import AutoTokenizer
        self.base_model = base_model
        self.cutoff = cutoff
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    def analyze_token_distribution(self, items: list[Item]) -> Tuple[float, int, int]:
        """Calculates avg tokens, max tokens, and count exceeding cutoff."""
        counts = [item.count_tokens(self.tokenizer) for item in items]
        avg_tokens = sum(counts) / len(counts) if counts else 0
        max_tokens = max(counts) if counts else 0
        exceeding = len([c for c in counts if c > self.cutoff])
        return avg_tokens, max_tokens, exceeding

    def prepare_dataset(self, train: list[Item], val: list[Item], test: list[Item]):
        """Formats prompt and completion fields across train, val, and test splits."""
        for item in tqdm(train + val, desc="Formatting train/val prompts"):
            item.make_prompts(self.tokenizer, self.cutoff, do_round=True)
        for item in tqdm(test, desc="Formatting test prompts"):
            item.make_prompts(self.tokenizer, self.cutoff, do_round=False)
