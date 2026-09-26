"""Which encoders the vLLM captioner lets the engine profile."""
from continuo_expressive.captioners import mm_limits


def test_text_only_captioning_hands_the_engine_no_modality_at_all():
    # each non-zero limit is an encoder vLLM pushes a dummy input through at startup
    assert mm_limits(text_only=True) == {"audio": 0, "image": 0, "video": 0}


def test_audio_captioning_keeps_audio_and_nothing_else():
    assert mm_limits(text_only=False) == {"audio": 1, "image": 0, "video": 0}


from continuo_expressive.captioners import select_fallback_kernels


def test_the_umbrella_switch_selects_every_fallback():
    env = {"CONTINUO_EXPRESSIVE_VLLM_NO_PREBUILT_KERNELS": "1"}
    select_fallback_kernels(env)
    assert env["CONTINUO_EXPRESSIVE_VLLM_ATTENTION"] == "TRITON_ATTN"
    assert env["CONTINUO_EXPRESSIVE_VLLM_MOE_BACKEND"] == "triton"
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"


def test_the_umbrella_switch_does_not_override_a_specific_choice():
    env = {"CONTINUO_EXPRESSIVE_VLLM_NO_PREBUILT_KERNELS": "1", "CONTINUO_EXPRESSIVE_VLLM_MOE_BACKEND": "cutlass"}
    select_fallback_kernels(env)
    assert env["CONTINUO_EXPRESSIVE_VLLM_MOE_BACKEND"] == "cutlass"


def test_off_by_default_touches_nothing():
    for env in ({}, {"CONTINUO_EXPRESSIVE_VLLM_NO_PREBUILT_KERNELS": "0"}):
        select_fallback_kernels(env)
        assert "CONTINUO_EXPRESSIVE_VLLM_ATTENTION" not in env


def test_max_new_tokens_reaches_the_captioner():
    from continuo_expressive.captioners import build_captioner, MAX_NEW_TOKENS
    c = build_captioner("vllm", model_id="x", max_new_tokens=1024)
    assert c.max_new_tokens == 1024 and c.n_cut == 0
    assert build_captioner("vllm", model_id="x").max_new_tokens == MAX_NEW_TOKENS


def test_generate_skips_a_prompt_that_cannot_fit(monkeypatch):
    from continuo_expressive import captioners as C

    class Out:  # the shape of vLLM's RequestOutput that generate reads
        def __init__(self, text):
            self.outputs = [type("O", (), {"text": text, "finish_reason": "stop"})()]

    class FakeLLM:
        def generate(self, requests, params):
            return [Out(f"cap for {r['prompt'][-3:]}") for r in requests]

    c = C.build_captioner("vllm", model_id="x", max_new_tokens=100)
    c.llm = FakeLLM()
    c._SamplingParams = lambda **kw: kw
    monkeypatch.setattr(c, "_templated", lambda prompt, with_audio: prompt)
    monkeypatch.setattr(c, "prompt_tokens", lambda t: C.MAX_MODEL_LEN if "BIG" in t else 10)
    got = c.generate([("a", None, "p-a"), ("b", None, "BIG-b"), ("c", None, "p-c")])
    assert got == ["cap for p-a", None, "cap for p-c"]
    assert c.n_skipped == 1
