import torch
from src.models.base import ModelAdapter, GenerationResult
from src.configs import ModelConfig, GenerationConfig
from src.tasks.base import ChatPrompt


def _find_layers(model) -> list:
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.model.model.layers,
    ]
    for getter in candidates:
        try:
            layers = getter(model)
            if layers:
                return list(layers)
        except AttributeError:
            continue
    raise RuntimeError("Could not locate transformer layer list in model")


def _find_hidden_dim(model) -> int:
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    return text_cfg.hidden_size


class Qwen35Adapter(ModelAdapter):
    def load_tokenizer(self, model_config: ModelConfig) -> None:
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_config.model_id)

    def load_for_generation(self, model_config: ModelConfig, gen_config: GenerationConfig, max_model_len: int) -> None:
        from vllm import LLM
        self._llm = LLM(
            model=model_config.model_id,
            tensor_parallel_size=gen_config.num_gpus,
            dtype=gen_config.dtype,
            max_model_len=max_model_len,
        )

    def load_for_extraction(self, model_config: ModelConfig, gen_config: GenerationConfig) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self._tokenizer = AutoTokenizer.from_pretrained(model_config.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_config.model_id,
            device_map="auto",
            torch_dtype=dtype_map[gen_config.dtype],
        )
        self._model.eval()
        self._n_layers = len(_find_layers(self._model))
        self._hidden_dim = _find_hidden_dim(self._model)

    def build_prompt(self, prompt: ChatPrompt) -> list[int]:
        messages = []
        if prompt.system_content is not None:
            messages.append({"role": "system", "content": prompt.system_content})
        messages.append({"role": "user", "content": prompt.user_content})
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        return self._tokenizer.encode(text, add_special_tokens=False)

    def generate(
        self,
        prompt_token_ids: list[list[int]],
        gen_config: GenerationConfig,
    ) -> list[GenerationResult]:
        from vllm import SamplingParams
        sampling_kwargs = {}
        if gen_config.repetition_penalty is not None:
            sampling_kwargs["repetition_penalty"] = gen_config.repetition_penalty
        if gen_config.presence_penalty is not None:
            sampling_kwargs["presence_penalty"] = gen_config.presence_penalty
        params = SamplingParams(
            temperature=gen_config.temperature,
            top_p=gen_config.top_p or 1.0,
            top_k=gen_config.top_k or -1,
            min_p=gen_config.min_p or 0.0,
            max_tokens=gen_config.max_new_tokens,
            **sampling_kwargs,
        )
        prompts = [{"prompt_token_ids": ids} for ids in prompt_token_ids]
        outputs = self._llm.generate(prompts, sampling_params=params)
        results = []
        for prompt_ids, out in zip(prompt_token_ids, outputs):
            gen_ids = list(out.outputs[0].token_ids)
            raw_text = self._tokenizer.decode(gen_ids, skip_special_tokens=False)
            results.append(GenerationResult(
                prompt_token_ids=prompt_ids,
                generated_token_ids=gen_ids,
                raw_text=raw_text,
            ))
        return results

    def extract_hidden_states(
        self,
        sequences: list[list[int]],
        prompt_lengths: list[int],
        layer_indices: list[int],
        stride: int,
    ) -> list[dict[int, torch.Tensor]]:
        device = next(self._model.parameters()).device

        max_len = max(len(s) for s in sequences)
        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        input_ids = torch.tensor(
            [[pad_id] * (max_len - len(s)) + s for s in sequences],
            dtype=torch.long, device=device,
        )
        attention_mask = (input_ids != pad_id).long()
        padding_lengths = [max_len - len(s) for s in sequences]

        with torch.no_grad():
            out = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        results = []
        for i, (pad_len, prompt_len) in enumerate(zip(padding_lengths, prompt_lengths)):
            gen_start = pad_len + prompt_len
            gen_end = max_len
            per_seq = {}
            for li in layer_indices:
                hs = out.hidden_states[li + 1]
                gen_hs = hs[i, gen_start:gen_end:stride, :].to(torch.float16).cpu()
                per_seq[li] = gen_hs
            results.append(per_seq)
        return results
