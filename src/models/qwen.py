import torch
from src.models.base import ModelAdapter, GenerationResult
from src.configs import ModelConfig, GenerationConfig
from src.tasks.base import ChatPrompt


class QwenAdapter(ModelAdapter):
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
        extraction_masks: list[list[int]] | None = None,
    ) -> list[dict[int, torch.Tensor]]:
        device = next(self._model.parameters()).device

        # Left-pad sequences to the same length
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

        # hidden_states: tuple of (n_layers+1) tensors, each [batch, seq, hidden]
        results = []
        if extraction_masks is not None:
            # Left-pad masks to match sequence padding
            padded_masks = torch.tensor(
                [[0] * (max_len - len(m)) + m for m in extraction_masks],
                dtype=torch.bool, device=device,
            )
            for i in range(len(sequences)):
                positions = padded_masks[i].nonzero(as_tuple=True)[0][::stride]
                per_seq = {}
                for li in layer_indices:
                    hs = out.hidden_states[li + 1]
                    per_seq[li] = hs[i, positions].to(torch.float16).cpu()
                results.append(per_seq)
        else:
            for i, (pad_len, prompt_len) in enumerate(zip(padding_lengths, prompt_lengths)):
                gen_start = pad_len + prompt_len
                per_seq = {}
                for li in layer_indices:
                    hs = out.hidden_states[li + 1]
                    gen_hs = hs[i, gen_start::stride].to(torch.float16).cpu()
                    per_seq[li] = gen_hs
                results.append(per_seq)
        return results
