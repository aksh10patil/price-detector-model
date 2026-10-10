from pricer.config import SUMMARIZATION_SYSTEM_PROMPT, DEFAULT_BATCH_MODEL


class Preprocessor:
    """Synchronous LLM preprocessor to summarize raw item text into uniform schema."""

    def __init__(self, model_name: str = DEFAULT_BATCH_MODEL, reasoning_effort: str = "low"):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort

    def messages_for(self, text: str) -> list[dict]:
        return [
            {"role": "system", "content": SUMMARIZATION_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

    def preprocess(self, text: str) -> str:
        from litellm import completion
        messages = self.messages_for(text)
        response = completion(
            messages=messages,
            model=self.model_name,
            reasoning_effort=self.reasoning_effort,
        )
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens
        if hasattr(response, "_hidden_params") and "response_cost" in response._hidden_params:
            self.total_cost += response._hidden_params["response_cost"]
        return response.choices[0].message.content
