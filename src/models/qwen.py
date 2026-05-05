import torch
from src.models.base import ModelAdapter
from src.configs import ModelConfig, HardwareConfig
from src.tasks.base import ChatPrompt

# Token ID for </think> in Qwen3 tokenizer
_THINK_END_TOKEN_ID = 151668
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"


class QwenAdapter(ModelAdapter):
    def load(self, model_config: ModelConfig, hardware_config: HardwareConfig) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map[hardware_config.dtype]
        self._model_config = model_config
        self._tokenizer = AutoTokenizer.from_pretrained(model_config.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_config.model_id,
            device_map=hardware_config.device_map,
            dtype=dtype,
        )
        self._model.eval()

    def get_layer_modules(self) -> list:
        return list(self._model.model.layers)

    def get_hidden_dim(self) -> int:
        return self._model.config.hidden_size

    def tokenize(self, prompt: "ChatPrompt") -> dict:
        messages = [{"role": "user", "content": prompt.user_content}]
        if prompt.assistant_prefill is not None:
            assistant_prefill = prompt.assistant_prefill + _MAGIC_SPLITTER_
            messages.append({"role": "assistant", "content": assistant_prefill})
            text = self._tokenizer.apply_chat_template(messages, tokenize=False).split(_MAGIC_SPLITTER_)[0]
        else:
            text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self._tokenizer([text], return_tensors="pt").to(self._model.device)

    def generate(
        self, inputs: dict, max_new_tokens: int, temperature: float,
        top_p: float | None = None, top_k: int | None = None, min_p: float | None = None,
    ) -> tuple[str, list[int]]:
        sampling_kwargs = {k: v for k, v in {"top_p": top_p, "top_k": top_k, "min_p": min_p}.items() if v is not None}
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                **sampling_kwargs,
            )
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output[0][input_len:].tolist()

        # Split thinking tokens from response tokens at </think>
        try:
            think_end = len(new_tokens) - new_tokens[::-1].index(_THINK_END_TOKEN_ID)
        except ValueError:
            think_end = 0

        content_tokens = new_tokens[think_end:]
        decoded = self._tokenizer.decode(content_tokens, skip_special_tokens=True).strip("\n")
        raw = self._tokenizer.decode(new_tokens, skip_special_tokens=False).strip("\n")
        return decoded, new_tokens, raw
