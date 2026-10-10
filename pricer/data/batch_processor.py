import os
import json
import pickle
from pathlib import Path
from functools import cache
from tqdm import tqdm
from pricer.config import SUMMARIZATION_SYSTEM_PROMPT, DEFAULT_BATCH_MODEL, JSONL_DIR
from pricer.data.schema import Item


@cache
def get_groq_client():
    from groq import Groq
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))


class BatchProcessor:
    """Manages asynchronous batch requests for large-scale product summarization."""

    BATCH_SIZE = 1_000

    def __init__(self, model: str = DEFAULT_BATCH_MODEL):
        self.model = model
        self.batches: list[dict] = []
        self.state_file = Path("batches.pkl")
        JSONL_DIR.mkdir(parents=True, exist_ok=True)

    def make_jsonl_row(self, item: Item) -> dict:
        return {
            "custom_id": str(item.id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SUMMARIZATION_SYSTEM_PROMPT},
                    {"role": "user", "content": item.full},
                ],
                "reasoning_effort": "low",
            },
        }

    def create_batch_files(self, items: list[Item]):
        self.batches = []
        for i in range(0, len(items), self.BATCH_SIZE):
            chunk = items[i : i + self.BATCH_SIZE]
            file_name = JSONL_DIR / f"batch_{i}_{i + len(chunk)}.jsonl"
            with open(file_name, "w", encoding="utf-8") as f:
                for item in chunk:
                    f.write(json.dumps(self.make_jsonl_row(item)) + "\n")
            self.batches.append({"file_name": str(file_name), "status": "created"})
        print(f"Created {len(self.batches)} batch file(s) in {JSONL_DIR}.")
        self.save_state()

    def submit_batches(self):
        client = get_groq_client()
        for batch in tqdm(self.batches, desc="Submitting batches"):
            if batch.get("status") == "created":
                with open(batch["file_name"], "rb") as f:
                    uploaded = client.files.create(file=f, purpose="batch")
                job = client.batches.create(
                    completion_window="24h",
                    endpoint="/v1/chat/completions",
                    input_file_id=uploaded.id,
                )
                batch["batch_id"] = job.id
                batch["status"] = "in_progress"
        self.save_state()

    def fetch_results(self, items: list[Item]):
        client = get_groq_client()
        items_by_id = {str(item.id): item for item in items}
        for batch in self.batches:
            if batch.get("batch_id"):
                res = client.batches.retrieve(batch["batch_id"])
                if res.output_file_id:
                    content = client.files.content(res.output_file_id).content.decode("utf-8")
                    for line in content.strip().split("\n"):
                        data = json.loads(line)
                        custom_id = data["custom_id"]
                        summary = data["response"]["body"]["choices"][0]["message"]["content"]
                        if custom_id in items_by_id:
                            items_by_id[custom_id].summary = summary
                    batch["status"] = "completed"
        self.save_state()

    def save_state(self):
        with open(self.state_file, "wb") as f:
            pickle.dump(self.batches, f)
