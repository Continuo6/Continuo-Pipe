"""Two ways to run the captioner behind one interface.

Both take ``(clip_id, waveform, prompt)`` triples and return one caption per triple, in
order. The difference is entirely throughput:

``transformers``
    One clip per ``generate()`` call. Simple, no extra dependency, and the reference
    for what the model outputs. It is also slow on this model for a structural reason:
    Qwen3-Omni-Captioner is a 30B mixture-of-experts, so every decoded token pulls the
    active experts' weights out of HBM, and at batch 1 that read is amortised over a
    single sequence. The GPU spends its time moving weights, not computing.

``vllm``
    Continuous batching: many sequences decode together, sharing each weight read, and
    finished sequences are replaced immediately instead of waiting for the slowest one
    in a fixed batch. Same model, same greedy decode, far better hardware utilisation.

The two are not bit-identical — different attention and MoE kernels, different
reduction orders — so a corpus captioned by both carries a mix of two greedy decodes of
the same model on the same prompts. That is a real if small inconsistency; prefer one
backend for a whole corpus where it matters.

vLLM returns a chunk's results only when the whole chunk is done, so the caller feeds
it in blocks and checkpoints between them. Blocks are what keeps a long run resumable;
they cost almost nothing as long as each block is large enough to keep the scheduler
saturated.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import sys

import numpy as np

from . import config
from .config import TARGET_SR

MAX_NEW_TOKENS = 448
MAX_MODEL_LEN = 8192


class Captioner(ABC):
    """Generate one caption per (clip_id, waveform, prompt)."""

    #: how many clips the driver should hand over at once
    chunk_size = 1

    def __init__(self, model_id: str, trust_remote_code: bool = False,
                 temperature: float = 0.0, max_new_tokens: int = MAX_NEW_TOKENS):
        self.model_id = model_id
        self.trust_remote_code = trust_remote_code
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        #: captions that ran into max_new_tokens so far (vLLM reports it; see generate)
        self.n_cut = 0
        #: prompts left uncaptioned because they could not fit the context
        self.n_skipped = 0

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def generate(self, items: Sequence[tuple[str, np.ndarray | None, str]]) -> list[str | None]:
        """-> one caption per item, or None where generation failed.

        A ``None`` waveform means caption from the prompt alone. That is how a long
        recording is captioned: it cannot be played to a model that takes 30 s, and an
        excerpt would have the model describe an arc it never heard.
        """


class TransformersCaptioner(Captioner):
    chunk_size = 1

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()          # keep info/warning noise out of the log

        # The top-level Qwen3-Omni no-split list in transformers 5.0 names an older
        # decoder class, so Accelerate can cut one decoder layer across two GPUs.
        # Keep each current thinker decoder layer intact while balancing whole layers.
        no_split = set(Qwen3OmniMoeForConditionalGeneration._no_split_modules or ())
        no_split.add("Qwen3OmniMoeThinkerTextDecoderLayer")
        Qwen3OmniMoeForConditionalGeneration._no_split_modules = sorted(no_split)

        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.model_id, dtype="auto", device_map="auto",
            trust_remote_code=self.trust_remote_code).eval()
        try:
            self.model.disable_talker()           # captioning never needs speech output
        except Exception:
            pass

        # Set pad_token_id up front so generate() stays quiet. Captions come from
        # batch_decode(skip_special_tokens=True) either way, so no special token can
        # reach the caption field regardless — this only keeps the run log clean.
        gen = self.model.generation_config
        if getattr(gen, "pad_token_id", None) is None:
            gen.pad_token_id = (getattr(gen, "eos_token_id", None)
                                or getattr(self.processor.tokenizer, "eos_token_id", None))

    def generate(self, items: Sequence[tuple[str, np.ndarray | None, str]]) -> list[str | None]:
        return [self._one(clip_id, wav, prompt) for clip_id, wav, prompt in items]

    def _one(self, clip_id: str, wav: np.ndarray | None, prompt: str) -> str:
        torch = self._torch
        from .prompts import generation_seed

        content = ([] if wav is None else [{"type": "audio", "audio": wav}])
        conversation = [{"role": "user", "content": content + [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True,
                                                  tokenize=False)
        kwargs = {} if wav is None else {"audio": [wav], "sampling_rate": TARGET_SR}
        inputs = self.processor(text=text, return_tensors="pt", padding=True,
                                **kwargs).to(self.model.device)
        model_dtype = next(self.model.parameters()).dtype
        for key in list(inputs.keys()):
            value = inputs[key]
            if hasattr(value, "dtype") and value.dtype == torch.float32:
                inputs[key] = value.to(model_dtype)

        sampling = (dict(do_sample=True, temperature=self.temperature, top_p=0.9)
                    if self.temperature > 0 else dict(do_sample=False))
        torch.manual_seed(generation_seed(clip_id))   # reproducible even when sampling
        with torch.no_grad():
            try:
                out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                          return_audio=False, **sampling)
            except TypeError:                          # older signature without return_audio
                out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, **sampling)
        ids = out[0] if isinstance(out, (tuple, list)) else out
        generated = ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def select_fallback_kernels(env) -> None:
    """CONTINUO_EXPRESSIVE_VLLM_NO_PREBUILT_KERNELS=1 -> the Triton/torch fallbacks for every kernel that
    may be unsupported on newer GPUs. Only fills variables the caller left unset, so the
    fine-grained ones still win, and does nothing unless the switch is on."""
    if env.get("CONTINUO_EXPRESSIVE_VLLM_NO_PREBUILT_KERNELS", "").strip() in ("", "0"):
        return
    env.setdefault("CONTINUO_EXPRESSIVE_VLLM_ATTENTION", "TRITON_ATTN")
    env.setdefault("CONTINUO_EXPRESSIVE_VLLM_MOE_BACKEND", "triton")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def mm_limits(text_only: bool) -> dict[str, int]:
    """Which modalities the engine may be handed, and therefore which encoders it
    profiles at startup. vLLM pushes a dummy input through every modality with a
    non-zero limit, so on a card that cannot run the bundled flash-attn PTX
    each unneeded encoder is a way to fail before the first prompt:
    the vision tower first, then — with image/video at 0 — the audio tower. Text-only
    captioning needs neither."""
    return {"audio": 0 if text_only else 1, "image": 0, "video": 0}


class VllmCaptioner(Captioner):
    """vLLM engine. Needs the model to be loadable by vLLM's Qwen3-Omni-MoE path.

    ``gpu_memory_utilization`` is the fraction of the device vLLM claims for weights
    plus KV cache; the default leaves a little headroom for the audio front end.

    ``tensor_parallel_size`` defaults to 1 because one card holding the whole model is
    both simplest and fastest when the card is big enough. It is not always: the 30B
    captioner is 60 GB in bf16 and does not fit a 24 GB card at any batch size. An
    earlier note here said the model's MoE ``grouped_mm`` path could not shard; newer
    vLLM versions can. Sharding on PCIe-only cards has no NVLink and vLLM disables its custom
    all-reduce past two of them, so expect the interconnect, not the GPU, to set the pace.
    """

    def __init__(self, *a, chunk_size: int = 256, gpu_memory_utilization: float = 0.90,
                 text_only: bool = False,
                 tensor_parallel_size: int = 1, **kw):
        super().__init__(*a, **kw)
        self.chunk_size = chunk_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.text_only = text_only

    def load(self) -> None:
        # vLLM must be imported before transformers, and its workers must spawn rather
        # than fork. transformers probes CUDA on import, and a process that has touched
        # CUDA cannot fork a child that initialises it again — the engine core dies with
        # "Cannot re-initialize CUDA in forked subprocess". Importing vllm first lets it
        # install its own start method before anything reaches the driver.
        import os
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        # For GPUs unsupported by prebuilt flash-attn or FlashInfer kernels, select
        # Triton or torch fallbacks. Fine-grained variables still work on their own.
        select_fallback_kernels(os.environ)
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        self._SamplingParams = SamplingParams
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code)
        # image/video are pinned to 0, not merely left unset: vLLM profiles every
        # modality the model *could* take, and for Qwen3-Omni that means pushing a dummy
        # video through the vision tower at startup. Captioning never sends one, and on
        # a card whose driver cannot JIT the bundled flash-attn PTX, that dummy pass
        # can fail. A zero limit skips it.
        extra = {}
        backend = os.environ.get("CONTINUO_EXPRESSIVE_VLLM_ATTENTION", "").strip()
        if backend:
            # Opt-in, for a card whose driver cannot run the bundled flash-attn PTX
            # ("compiled with an unsupported toolchain"): TRITON_ATTN is JIT-compiled
            # for the card it runs on and needs no prebuilt kernel. Not the default —
            # the flash path is faster wherever it works. Set for the decoder and, when
            # this vLLM exposes it, for the multimodal encoders, which pick their own.
            import inspect
            from vllm.config.attention import AttentionConfig
            from vllm.v1.attention.backends.registry import AttentionBackendEnum
            extra["attention_config"] = AttentionConfig(backend=AttentionBackendEnum[backend])
            if "mm_encoder_attn_backend" in inspect.signature(LLM.__init__).parameters \
                    or "kwargs" in inspect.signature(LLM.__init__).parameters:
                extra["mm_encoder_attn_backend"] = backend
        moe = os.environ.get("CONTINUO_EXPRESSIVE_VLLM_MOE_BACKEND", "").strip()
        if moe:
            # Same box, next kernel down: the MoE experts default to FlashInfer's CUTLASS
            # kernels, which ship no sm_120 build and try to JIT one — impossible with
            # the box's nvcc 12.5 ("No supported CUDA architectures found for major
            # versions [12]"). `triton` is the pure-Triton fused MoE, compiled for the
            # card by Triton itself. Passed straight through to EngineArgs.moe_backend.
            extra["moe_backend"] = moe
        self.llm = LLM(model=self.model_id, trust_remote_code=self.trust_remote_code,
                       tensor_parallel_size=self.tensor_parallel_size,
                       limit_mm_per_prompt=mm_limits(self.text_only),
                       max_model_len=MAX_MODEL_LEN,
                       gpu_memory_utilization=self.gpu_memory_utilization, **extra)

    def _templated(self, prompt: str, with_audio: bool = True) -> str:
        content = ([{"type": "audio", "audio": "placeholder.wav"}] if with_audio else [])
        conversation = [{"role": "user", "content": content + [{"type": "text", "text": prompt}]}]
        return self.processor.apply_chat_template(conversation, add_generation_prompt=True,
                                                  tokenize=False)

    def prompt_tokens(self, templated: str) -> int:
        """Token count of a templated prompt, with the engine's own tokenizer."""
        if getattr(self, "_tok", None) is None:
            self._tok = self.llm.get_tokenizer()
        return len(self._tok.encode(templated))

    def generate(self, items: Sequence[tuple[str, np.ndarray, str]]) -> list[str | None]:
        from .prompts import generation_seed

        requests, params, sent, skipped = [], [], [], []
        for k, (clip_id, wav, prompt) in enumerate(items):
            request = {"prompt": self._templated(prompt, with_audio=wav is not None)}
            # vLLM validates the whole batch and raises on the first prompt that cannot
            # fit its context. Count the tokens first and leave such a clip uncaptioned (the
            # audio's own tokens are not counted here; text-only runs are exact).
            n = self.prompt_tokens(request["prompt"])
            if n + self.max_new_tokens > MAX_MODEL_LEN:
                skipped.append((clip_id, n))
                continue
            if wav is not None:
                request["multi_modal_data"] = {"audio": [(wav, TARGET_SR)]}
            requests.append(request)
            sent.append(k)
            params.append(self._SamplingParams(
                max_tokens=self.max_new_tokens, temperature=self.temperature,
                top_p=0.9 if self.temperature > 0 else 1.0,
                seed=generation_seed(clip_id) if self.temperature > 0 else None))
        if skipped:
            self.n_skipped += len(skipped)
            print(f"[warn] {len(skipped)} prompt(s) too long for the {MAX_MODEL_LEN}-token "
                  f"context with {self.max_new_tokens} reserved for the answer; left "
                  f"uncaptioned, e.g. {[(c, n) for c, n in skipped[:3]]}",
                  file=sys.stderr, flush=True)

        outputs = self.llm.generate(requests, params) if requests else []
        captions: list[str | None] = [None] * len(items)
        cut = []
        for k, out in zip(sent, outputs):
            clip_id = items[k][0]
            text = out.outputs[0].text if out.outputs else None
            captions[k] = text.strip() if text else None
            if out.outputs and out.outputs[0].finish_reason == "length":
                cut.append(clip_id)
        if cut:
            # a caption that ran into max_tokens ends mid-sentence; on a dialogue it means a
            # speaker is missing. Nothing else records this, so say so where the log is read.
            self.n_cut += len(cut)
            print(f"[warn] {len(cut)} caption(s) hit max_new_tokens={self.max_new_tokens} and "
                  f"were cut off, e.g. {cut[:3]}", file=sys.stderr, flush=True)
        return captions


def build_captioner(backend: str, model_id: str = "", trust_remote_code: bool = False,
                    temperature: float = 0.0, chunk_size: int = 256,
                    gpu_memory_utilization: float = 0.90,
                    tensor_parallel_size: int = 1, text_only: bool = False,
                    max_new_tokens: int = MAX_NEW_TOKENS) -> Captioner:
    model_id = model_id or config.captioner_model()
    if backend == "vllm":
        return VllmCaptioner(model_id, trust_remote_code, temperature,
                             max_new_tokens=max_new_tokens,
                             chunk_size=chunk_size,
                             gpu_memory_utilization=gpu_memory_utilization,
                             tensor_parallel_size=tensor_parallel_size,
                             text_only=text_only)
    if backend == "transformers":
        return TransformersCaptioner(model_id, trust_remote_code, temperature,
                                     max_new_tokens=max_new_tokens)
    raise ValueError(f"unknown backend {backend!r}; expected 'vllm' or 'transformers'")
