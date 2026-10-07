import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from pricer.data.parser import parse
from pricer.data.schema import Item

CHUNK_SIZE = 1000
CPU_COUNT = os.cpu_count() or 4
WORKERS = max(CPU_COUNT - 1, 1)


class ItemLoader:
    """Loads and parallel-parses Amazon Review dataset records."""

    def __init__(self, category: str):
        self.category = category
        self.dataset = None

    def from_datapoint(self, datapoint: dict) -> Item | None:
        return parse(datapoint, self.category)

    def from_chunk(self, chunk: list[dict]) -> list[Item]:
        batch = [self.from_datapoint(dp) for dp in chunk]
        return [item for item in batch if item is not None]

    def chunk_generator(self):
        chunk = []
        for datapoint in self.dataset:
            chunk.append(datapoint)
            if len(chunk) == CHUNK_SIZE:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    def load(self, limit: int | None = None) -> list[Item]:
        from datasets import load_dataset
        print(f"Loading raw Amazon Reviews for category '{self.category}'...")
        self.dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{self.category}", split="full")
        items = []

        total_chunks = len(self.dataset) // CHUNK_SIZE + 1
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            for chunk_items in tqdm(executor.map(self.from_chunk, self.chunk_generator()), total=total_chunks):
                items.extend(chunk_items)
                if limit and len(items) >= limit:
                    items = items[:limit]
                    break

        print(f"Loaded and parsed {len(items):,} clean items for {self.category}.")
        return items
