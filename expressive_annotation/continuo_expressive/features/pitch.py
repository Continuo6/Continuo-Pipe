"""Pitch = mean-over-all-frames F0 via penn (fcnf0++), bucketed per gender.

penn is the estimator DataSpeech ran to produce ``utterance_pitch_mean`` — the exact
field the ParaSpeechCaps pitch edges were tertiled from — so measuring with penn is
what puts ``pitch_hz`` on the same scale as the edges in :mod:`.buckets`. pyin reads
male voices ~10-15 Hz high against those edges and has ~13% gross-error on short
clips versus penn's ~1%.

Aggregation is the **mean over all frames**, matching how ``utterance_pitch_mean`` is
defined — not the median of voiced frames. That choice is worth about
Pearson r 0.80 -> 0.88 against the PSC reference and leaves the absolute scale
essentially unbiased (male 161.5 Hz vs reference 161.8).

On per-clip failure the field stays null; there is deliberately **no pyin fallback**,
since dropping pyin's gross errors was the point of moving to penn.
"""
from __future__ import annotations

import sys
from typing import Sequence

import numpy as np

from ..config import TARGET_SR
from .buckets import PITCH_LABELS, bucket, pitch_edges_for

MIN_SECONDS = 0.5
FMIN, FMAX = 65.0, 500.0
HOP_SECONDS = 160 / TARGET_SR   # 10 ms
#: frames per forward pass in :func:`measure_batch`. Frames are independent, so this
#: only bounds memory: 4096 x 1440 logits in fp32 is about 24 MB.
INFER_FRAMES = 4096

_warned = False


def measure(wav: np.ndarray | None, gender: str | None = None, sr: int = TARGET_SR,
            gpu: int | None = 0) -> dict:
    """-> ``{"pitch_hz": float|None, "pitch": "low"|"medium"|"high"|None}``.

    ``gpu`` is a device index for penn, or None for CPU.
    """
    # One clip is a batch of one. This used to call penn.from_audio directly, which
    # decodes with torbi's compiled kernel whatever the card — so on a card that kernel
    # cannot run on (see _viterbi_backend) the single-clip probe failed while the batch
    # path the workers use had a fallback, or the other way round. One path, one answer.
    return measure_batch([wav], [gender], sr=sr, gpu=gpu)[0]


def _summarise(f0: np.ndarray, gender: str | None) -> dict:
    """Voiced-frame mean and its bucket, from one clip's frames."""
    out = {"pitch_hz": None, "pitch": None}
    voiced = f0[np.isfinite(f0) & (f0 > 0)]
    if voiced.size:
        hz = round(float(np.mean(voiced)), 1)
        out["pitch_hz"] = hz
        out["pitch"] = bucket(hz, pitch_edges_for(gender), PITCH_LABELS)
    return out


def viterbi_threads() -> int:
    """CPU threads for the Viterbi decode: ``CONTINUO_EXPRESSIVE_VITERBI_THREADS``, else torch's count."""
    import os
    import torch
    raw = os.environ.get("CONTINUO_EXPRESSIVE_VITERBI_THREADS", "").strip()
    if raw:
        return max(1, int(raw))
    return max(1, torch.get_num_threads())


_BACKEND: dict[str, str] = {}


def _viterbi_backend(device: str) -> str:
    """``"torbi"`` or ``"torch"`` for this device: ``CONTINUO_EXPRESSIVE_VITERBI_BACKEND`` if set, else
    torbi on CPU and, on a GPU, torbi if its compiled kernel runs there.

    Some torbi builds lack a kernel for newer GPU architectures. The probe is one
    tiny decode, once per process per device.
    """
    import os
    forced = os.environ.get("CONTINUO_EXPRESSIVE_VITERBI_BACKEND", "").strip().lower()
    if forced in ("torbi", "torch"):
        return forced
    if device == "cpu":
        return "torbi"
    if device not in _BACKEND:
        import torch
        import torbi
        try:
            torbi.from_probabilities(
                observation=torch.full((1, 2, 4), 0.25, device=device),
                batch_frames=torch.tensor([2]),
                gpu=int(device.split(":")[1]))
            _BACKEND[device] = "torbi"
        except Exception as e:  # RuntimeError from the op, whatever its wording
            print(f"[pitch] torbi's Viterbi kernel cannot run on {device} "
                  f"({type(e).__name__}: {str(e).splitlines()[0][:80]}); decoding with "
                  "torch ops on the same device instead", file=sys.stderr, flush=True)
            _BACKEND[device] = "torch"
    return _BACKEND[device]


_BAND: dict[tuple, tuple] = {}


def _viterbi_torch(observation, widths: Sequence[int], transition, initial, device: str):
    """torbi.from_probabilities in plain torch ops: the same log-domain Viterbi, batched,
    each row decoded against its own length. Returns bins ``(N, T)``.

    penn's transition is banded — a state can move at most W bins between frames, W=77
    here — so the max over predecessors is taken over the 2W+1 band rather than all
    1440 states, and the band table is built once per (device, matrix). Inputs go
    through the same log/exp/tiny/log sequence torbi applies, so the scores it compares
    are the same numbers.
    """
    import torch
    N, T, S = observation.shape
    key = (device, id(transition))
    if key not in _BAND:
        logT = torch.log(transition.to(device=device, dtype=torch.float32))
        rows, cols = torch.nonzero(transition > 0, as_tuple=True)
        W = int(max(int((cols - rows).max()), int((rows - cols).max())))
        # band[s, k] = logT[s - W + k, s]: the score of arriving at s from s - W + k
        padded = torch.full((S, S + 2 * W), -float("inf"), device=device)
        padded[:, W:W + S] = logT.T
        band = padded.unfold(1, 2 * W + 1, 1)[torch.arange(S), torch.arange(S)].contiguous()
        _BAND[key] = (band, W, torch.log(initial.to(device=device, dtype=torch.float32)))
    band, W, log_init = _BAND[key]

    # exactly torbi's preprocessing, in its order: the log is taken at the observation's
    # own precision (fp16 here, see measure_batch) *before* the cast to fp32 — taking
    # it in fp32 is a different, slightly better decode (the same trap measure_batch
    # already documents for the observation buffer)
    obs = torch.log(observation.to(device=device)).to(torch.float32)
    torch.exp_(obs)
    obs += torch.finfo(torch.float32).tiny
    torch.log_(obs)

    last = torch.tensor([w - 1 for w in widths], device=device)
    score = log_init[None] + obs[:, 0]                # (N, S)
    final = score.clone()
    back = torch.empty((T, N, S), dtype=torch.int16, device=device)
    neg = torch.full((N, W), -float("inf"), device=device)
    for t in range(1, T):
        cand = torch.cat([neg, score, neg], dim=1).unfold(1, 2 * W + 1, 1) + band[None]
        best, arg = cand.max(dim=2)
        score = best + obs[:, t]
        back[t] = arg.to(torch.int16)
        hit = last == t
        if bool(hit.any()):
            final[hit] = score[hit]
    state = final.argmax(dim=1)                       # (N,)
    bins = torch.zeros((N, T), dtype=torch.int64, device=device)
    bins[torch.arange(N), last] = state
    for t in range(T - 1, 0, -1):
        active = last >= t
        prev = state + back[t].gather(1, state[:, None]).squeeze(1).to(torch.int64) - W
        state = torch.where(active, prev, state)
        bins[:, t - 1] = torch.where(active, state, bins[:, t - 1])
    return bins


def measure_batch(wavs: Sequence[np.ndarray | None], genders: Sequence[str | None],
                  sr: int = TARGET_SR, gpu: int | None = 0) -> list[dict]:
    """:func:`measure` over a batch, sharing one pass through the network.

    penn's own interface takes a single signal, and on clips of a few seconds almost
    all of the cost is per call rather than per sample: 105 ms a clip against 7 ms when
    a batch goes through together. Pitch is where this pass spends most of its wall
    clock, so that is the difference between an hour and a day on a corpus.

    What makes it safe is where penn's stages couple. Framing and inference are
    per-frame — a frame's logits do not depend on its neighbours — so frames from many
    clips can ride through the network in one tensor. Only the Viterbi decoder walks a
    sequence, and it runs per clip on that clip's own logits, exactly as before. The
    results are identical, not merely close, which ``tests/test_pitch_batch.py`` checks
    against real audio.

    Concatenating the *waveforms* instead would be simpler and wrong: one signal means
    one Viterbi path, so neighbouring clips bleed into each other (measured: F0 off by
    up to 21.8 Hz, a pitch band changing in one clip in twenty) and memory grows with
    the whole batch rather than a fixed window.
    """
    out: list[dict] = [{"pitch_hz": None, "pitch": None} for _ in wavs]
    usable = [i for i, w in enumerate(wavs)
              if w is not None and len(w) >= sr * MIN_SECONDS]
    if not usable:
        return out

    global _warned
    try:
        import penn
        import torch

        device = "cpu" if gpu is None else f"cuda:{gpu}"
        frames, spans, cursor = [], {}, 0
        for i in usable:
            wav = np.ascontiguousarray(wavs[i], dtype=np.float32)
            chunks = list(penn.preprocess(torch.tensor(wav)[None].float(), sr,
                                          HOP_SECONDS, None, "half-window"))
            if not chunks:
                continue
            block = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
            spans[i] = (cursor, cursor + block.shape[0])
            cursor += block.shape[0]
            frames.append(block)
        if not spans:
            return out

        stacked = torch.cat(frames)
        logits = torch.cat([
            penn.infer(stacked[a:a + INFER_FRAMES].to(device)).detach()
            for a in range(0, stacked.shape[0], INFER_FRAMES)])

        # penn.postprocess masks bins outside [fmin, fmax] to -inf before decoding, and
        # skipping it is not a rounding difference: it lets the decoder wander outside
        # the band and moved F0 by up to 18.4 Hz on real clips.
        minidx = penn.convert.frequency_to_bins(torch.tensor(FMIN))
        maxidx = penn.convert.frequency_to_bins(torch.tensor(FMAX), torch.ceil)
        logits[:, :minidx] = -float("inf")
        logits[:, maxidx:] = -float("inf")

        # One Viterbi call for the whole batch. torbi decodes each row against its own
        # `batch_frames` length, so the rows stay independent — this is a batched
        # execution of the same per-clip decode, not a joint decode over a longer
        # sequence. Decoding is 86% of this pass's cost, and penn's own entry point
        # runs it one clip at a time.
        import torbi
        decoder = penn.decode.Viterbi()
        order = list(spans)
        widths = [spans[i][1] - spans[i][0] for i in order]
        longest = max(widths)
        # dtype follows the network's own output (fp16), because torbi decodes at the
        # precision it is handed: an fp32 buffer here is a *different*, slightly better
        # decode than penn's, and differed from it on 0.1% of clips. Matching penn is
        # what makes this batch path bit-identical to the per-clip one.
        observation = torch.zeros(len(order), longest, penn.PITCH_BINS,
                                  dtype=logits.dtype, device=device)
        for k, i in enumerate(order):
            lo, hi = spans[i]
            observation[k, :hi - lo] = torch.nn.functional.softmax(
                logits[lo:hi], dim=1).squeeze(-1)
        # torbi decodes on ONE thread unless told otherwise, and on CPU the decode is the
        # whole cost when run single-threaded. If torbi decodes on CPU, hand it the
        # threads the process already has. Deterministic either way — same bins.
        if _viterbi_backend(device) == "torbi":
            bins = torbi.from_probabilities(
                observation=observation,
                batch_frames=torch.tensor(widths),
                transition=decoder.transition,
                initial=decoder.initial,
                gpu=None if device == "cpu" else int(device.split(":")[1]),
                num_threads=viterbi_threads())
        else:
            bins = _viterbi_torch(observation, widths, decoder.transition,
                                  decoder.initial, device)

        for k, i in enumerate(order):
            lo, hi = spans[i]
            row = bins[k, :hi - lo].reshape(1, -1)
            pitch = penn.decode.local_expected_value_from_bins(
                row.T, logits[lo:hi]).T
            out[i] = _summarise(pitch.squeeze().cpu().numpy(), genders[i])
    except Exception as e:
        if not _warned:
            print(f"[pitch] penn F0 failed ({type(e).__name__}: {e}); pitch left null. "
                  "Install into this env: pip install penn", file=sys.stderr)
            _warned = True
    return out
