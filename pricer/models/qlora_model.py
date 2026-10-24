import re
from pricer.config import DEFAULT_BASE_MODEL


class QLoRAPricer:
    """Loads a 4-bit quantized base LLM with fine-tuned LoRA adapters for price inference."""

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        adapter_path: str | None = None,
        device: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        self.base_model_name = base_model
        self.adapter_path = adapter_path

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant_config if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        if adapter_path:
            self.model = PeftModel.from_pretrained(base, adapter_path)
        else:
            self.model = base

        if not torch.cuda.is_available() and self.device != "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()

    @staticmethod
    def extract_price(text: str) -> float:
        cleaned = text.replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", cleaned)
        return float(match.group()) if match else 0.0

    def predict(self, prompt: str) -> float:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=8)
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, prompt_len:]
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self.extract_price(decoded)

    def __call__(self, item) -> float:
        prompt = item["prompt"] if isinstance(item, dict) else (item.prompt or item.full)
        return self.predict(prompt)
