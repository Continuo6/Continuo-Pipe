"""``measure_batch`` must agree with ``measure``, clip for clip.

The batch path reimplements what ``penn.from_audio`` does — frame, infer, mask the
band, Viterbi-decode, take the local expected value — so that the decode can run for a
whole batch at once. That is only worth having if it is the *same* computation, so the
first test here runs both paths over the same signals and demands equality rather than
closeness. Two things were caught this way and neither was a rounding difference:

* skipping penn's fmin/fmax masking let the decoder wander outside the band (F0 off by
  up to 18.4 Hz);
* handing torbi an fp32 observation buffer when penn hands it fp16 is a different,
  slightly better decode, and disagreed with penn on 0.1% of clips.

That test needs penn and its checkpoint, so it skips where they are absent. The
bookkeeping tests below — order, and which slots stay null — need neither.
"""
from __future__ import annotations

import numpy as np
import pytest

from continuo_expressive.config import TARGET_SR
from continuo_expressive.features import pitch as pf


def _tone(hz: float, seconds: float, sr: int = TARGET_SR) -> np.ndarray:
    """A vibrato'd tone: penn tracks it, and it is the same every run."""
    t = np.arange(int(seconds * sr)) / sr
    f = hz * (1.0 + 0.03 * np.sin(2 * np.pi * 1.5 * t))
    phase = 2 * np.pi * np.cumsum(f) / sr
    wav = 0.5 * np.sin(phase) + 0.2 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
    return wav.astype(np.float32)


def _has_penn() -> bool:
    try:
        import penn  # noqa: F401
        import torbi  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _has_penn(), reason="penn/torbi not installed in this env")
def test_batch_matches_per_clip_exactly():
    # deliberately ragged, so rows get padded to the longest and short rows must not
    # read into the padding
    wavs = [_tone(hz, secs) for hz, secs in
            [(110, 1.3), (220, 0.7), (155, 2.1), (98, 1.0), (330, 1.6)]]
    genders = ["male", "female", "male", "male", "female"]

    one = [pf.measure(w, g, gpu=None) for w, g in zip(wavs, genders)]
    batched = pf.measure_batch(wavs, genders, gpu=None)

    assert [r["pitch_hz"] for r in batched] == [r["pitch_hz"] for r in one]
    assert [r["pitch"] for r in batched] == [r["pitch"] for r in one]
    assert all(r["pitch_hz"] is not None for r in one), "tones should track"


@pytest.mark.skipif(not _has_penn(), reason="penn/torbi not installed in this env")
def test_batch_size_does_not_change_the_answer():
    """Splitting the same clips differently must give the same numbers.

    This is the property ``--batch-age-head`` does *not* have, and the reason pitch can
    be batched by default: nothing here normalises across the batch.
    """
    wavs = [_tone(90 + 20 * i, 0.8 + 0.2 * (i % 4)) for i in range(6)]
    genders = ["male"] * 6

    whole = pf.measure_batch(wavs, genders, gpu=None)
    in_twos: list[dict] = []
    for a in range(0, 6, 2):
        in_twos += pf.measure_batch(wavs[a:a + 2], genders[a:a + 2], gpu=None)
    singly = [pf.measure_batch([w], ["male"], gpu=None)[0] for w in wavs]

    assert [r["pitch_hz"] for r in in_twos] == [r["pitch_hz"] for r in whole]
    assert [r["pitch_hz"] for r in singly] == [r["pitch_hz"] for r in whole]


def test_unusable_clips_keep_their_slot():
    """None and too-short clips stay null *in place*, so results still line up.

    The batch is compacted before it goes through the network, so a dropped clip is an
    off-by-one waiting to happen: every attribute is joined by position.
    """
    short = np.zeros(int(TARGET_SR * pf.MIN_SECONDS) - 1, dtype=np.float32)
    wavs = [None, short, None]
    out = pf.measure_batch(wavs, ["male", "female", None], gpu=None)

    assert len(out) == 3
    assert all(r == {"pitch_hz": None, "pitch": None} for r in out)


@pytest.mark.skipif(not _has_penn(), reason="penn/torbi not installed in this env")
def test_usable_and_unusable_mixed():
    """A null in the middle must not shift the clips after it."""
    good = _tone(120, 1.2)
    out = pf.measure_batch([good, None, good], ["male", "male", "male"], gpu=None)

    assert out[1] == {"pitch_hz": None, "pitch": None}
    assert out[0]["pitch_hz"] is not None
    assert out[0] == out[2], "the same signal twice must measure the same"


def test_torch_viterbi_matches_torbi_kernel():
    """The torch-op decode used where torbi's kernel cannot run must be the same decode."""
    torch = pytest.importorskip("torch")
    torbi = pytest.importorskip("torbi")
    penn = pytest.importorskip("penn")
    dec = penn.decode.Viterbi()
    g = torch.Generator().manual_seed(7)
    N, T, S = 3, 120, penn.PITCH_BINS
    # peaky distributions that wander, like a pitch track; fp16 like the network output
    centre = 400 + (torch.cumsum(torch.randn(N, T, generator=g), dim=1) * 6).long().clamp(60, S - 60)
    logits = -((torch.arange(S)[None, None] - centre[..., None]).float() ** 2) / 50
    logits += torch.randn(N, T, S, generator=g) * 0.5
    obs = torch.softmax(logits, dim=-1).half()
    widths = [T, T - 17, T - 40]
    torch.set_num_threads(4)
    want = torbi.from_probabilities(observation=obs, batch_frames=torch.tensor(widths),
                                    transition=dec.transition, initial=dec.initial,
                                    gpu=None, num_threads=4)
    got = pf._viterbi_torch(obs, widths, dec.transition, dec.initial, "cpu")
    for k, w in enumerate(widths):
        assert torch.equal(want[k, :w].long(), got[k, :w].long()), f"row {k} differs"


def test_viterbi_backend_honours_the_override(monkeypatch):
    monkeypatch.setenv("CONTINUO_EXPRESSIVE_VITERBI_BACKEND", "torch")
    assert pf._viterbi_backend("cuda:0") == "torch"
    monkeypatch.setenv("CONTINUO_EXPRESSIVE_VITERBI_BACKEND", "torbi")
    assert pf._viterbi_backend("cuda:0") == "torbi"
    monkeypatch.delenv("CONTINUO_EXPRESSIVE_VITERBI_BACKEND")
    assert pf._viterbi_backend("cpu") == "torbi"
