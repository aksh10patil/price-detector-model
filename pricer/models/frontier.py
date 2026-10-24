import re
from pricer.data.schema import Item


class FrontierPricer:
    """Zero-shot price estimation using commercial frontier LLMs via LiteLLM."""

    def __init__(self, model_name: str = "openai/gpt-5-nano", reasoning_effort: str | None = None, seed: int = 42):
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.seed = seed

    def messages_for(self, item: Item) -> list[dict]:
        text = item.summary or item.full
        msg = f"Estimate the price of this product. Respond with only the price as a number, no explanation\n\n{text}"
        return [{"role": "user", "content": msg}]

    @staticmethod
    def extract_number(response_text: str) -> float:
        cleaned = response_text.replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", cleaned)
        return float(match.group()) if match else 0.0

    def predict(self, item: Item) -> float:
        from litellm import completion
        kwargs = {"model": self.model_name, "messages": self.messages_for(item), "seed": self.seed}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        try:
            res = completion(**kwargs)
            content = res.choices[0].message.content
            return self.extract_number(content)
        except Exception as e:
            print(f"Frontier API error ({self.model_name}): {e}")
            return 50.0

    def __call__(self, item: Item) -> float:
        return self.predict(item)
