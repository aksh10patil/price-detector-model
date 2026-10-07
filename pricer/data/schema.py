from typing import Optional, Self
from pydantic import BaseModel
try:
    from datasets import load_dataset, DatasetDict, Dataset
except ImportError:
    load_dataset = None
    DatasetDict = None
    Dataset = None
from pricer.config import QUESTION, PREFIX, DEFAULT_LITE_DATASET


class Item(BaseModel):
    """Pydantic representation of an e-commerce product."""

    title: str
    price: float
    category: str
    features: Optional[list[str]] = None
    description: Optional[list[str]] = None
    details: Optional[dict[str, str]] = None
    weight: Optional[float] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    completion: Optional[str] = None
    id: Optional[int] = None

    def make_prompt(self, text: str):
        """Constructs an inference prompt from raw description text."""
        self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}"

    def count_tokens(self, tokenizer) -> int:
        """Count tokens in the summary using a Hugging Face tokenizer."""
        text = self.summary or self.full
        return len(tokenizer.encode(text, add_special_tokens=False))

    def make_prompts(self, tokenizer, max_tokens: int, do_round: bool = False):
        """Format training prompt and completion with token truncation."""
        text = self.summary or self.full
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_tokens:
            summary = tokenizer.decode(tokens[:max_tokens]).rstrip()
        else:
            summary = text
        self.prompt = f"{QUESTION}\n\n{summary}\n\n{PREFIX}"
        self.completion = f"{round(self.price)}.00" if do_round else str(self.price)

    def count_prompt_tokens(self, tokenizer) -> int:
        """Count total tokens in prompt and completion combined."""
        full = (self.prompt or "") + (self.completion or "")
        return len(tokenizer.encode(full, add_special_tokens=False))

    def to_datapoint(self) -> dict:
        """Convert item to SFT prompt/completion dictionary."""
        return {"prompt": self.prompt, "completion": self.completion}

    @property
    def full(self) -> str:
        """Construct raw full text representation before summarization."""
        text = f"Title: {self.title}\n"
        text += f"Category: {self.category}\n"
        if self.features:
            text += f"Features: {'; '.join(self.features)}\n"
        if self.description:
            text += f"Description: {' '.join(self.description)}\n"
        if self.details:
            details_str = ", ".join([f"{k}: {v}" for k, v in self.details.items()])
            text += f"Details: {details_str}\n"
        return text.strip()

    @classmethod
    def from_hub(cls, dataset_name: str = DEFAULT_LITE_DATASET) -> tuple[list[Self], list[Self], list[Self]]:
        """Load train, val, and test splits from Hugging Face Hub."""
        ds = load_dataset(dataset_name)
        train = [cls.model_validate(row) for row in ds["train"]]
        val = [cls.model_validate(row) for row in ds["validation"]]
        test = [cls.model_validate(row) for row in ds["test"]]
        return train, val, test

    @classmethod
    def push_to_hub(cls, dataset_name: str, train: list[Self], val: list[Self], test: list[Self]):
        """Push Item objects to Hugging Face Hub."""
        DatasetDict(
            {
                "train": Dataset.from_list([item.model_dump() for item in train]),
                "validation": Dataset.from_list([item.model_dump() for item in val]),
                "test": Dataset.from_list([item.model_dump() for item in test]),
            }
        ).push_to_hub(dataset_name)

    @classmethod
    def push_prompts_to_hub(cls, dataset_name: str, train: list[Self], val: list[Self], test: list[Self]):
        """Push prompt/completion pairs to Hugging Face Hub for SFT training."""
        DatasetDict(
            {
                "train": Dataset.from_list([item.to_datapoint() for item in train]),
                "val": Dataset.from_list([item.to_datapoint() for item in val]),
                "test": Dataset.from_list([item.to_datapoint() for item in test]),
            }
        ).push_to_hub(dataset_name)
