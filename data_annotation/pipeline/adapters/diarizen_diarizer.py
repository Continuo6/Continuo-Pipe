"""Adapter: DiariZen speaker diarization.

DiariZen's pipeline expects a file path only for the initial ``torchaudio.load``.
Everything after that runs on an in-memory ``{"waveform": tensor, "sample_rate": int}``.
This adapter mirrors that inference path directly on the pipeline's waveform to
avoid writing temporary WAVs.
"""

from __future__ import annotations

import numpy as np
import torch

from pipeline.config import DiariZenParams
from pipeline.stages.base import SpeakerDiarizer
from pipeline.types import AudioBundle, DiarizationFrame

class DiariZenDiarizer(SpeakerDiarizer):
    def __init__(self, params: DiariZenParams, device: str):
        self._params = params
        # Kept for adapter interface compatibility; DiariZen uses cuda:0 internally.
        self._device = device
        self._pipeline: object | None = None
        self._binarizer = None

    def warmup(self) -> None:
        if self._pipeline is not None:
            return
        # Required so pyannote 3.x weights load under PyTorch 2.6+.
        from pipeline._torch_compat import legacy_torch_load_context

        with legacy_torch_load_context():
            # Import may initialise lightning loaders, so keep it in scope too.
            from diarizen.pipelines.inference import DiariZenPipeline
            self._pipeline = DiariZenPipeline.from_pretrained(
                repo_id=self._params.model,
                **({"cache_dir": self._params.cache_dir}
                   if self._params.cache_dir else {}),
            )

        # Runtime overrides: both are read by get_embeddings/get_segmentations.
        self._pipeline.embedding_batch_size = self._params.embedding_batch_size
        self._pipeline.segmentation_batch_size = self._params.segmentation_batch_size

    def diarize(self, audio: AudioBundle) -> list[DiarizationFrame]:
        if self._pipeline is None:
            raise RuntimeError("call warmup() before diarize()")
        annotation = self._diarize_inmemory(audio.waveform, audio.sample_rate)
        return [
            DiarizationFrame(
                start=float(seg.start),
                end=float(seg.end),
                speaker=str(speaker),
            )
            for seg, _track, speaker in annotation.itertracks(yield_label=True)
        ]

    def _diarize_inmemory(self, waveform: np.ndarray, sample_rate: int):
        """Run the pipeline on an in-memory waveform, bypassing torchaudio.load.

        Mirrors :meth:`DiariZenPipeline.__call__` minus the file IO. Bound to
        a few of the library's internals (``get_segmentations``,
        ``get_embeddings``, ``clustering``, ``reconstruct``) — these match
        pyannote's stable inference primitives, so the surface is small.
        """
        # Lazy imports so importing this adapter doesn't pull in heavy deps.
        from pyannote.audio.utils.signal import Binarize

        pipe = self._pipeline
        assert pipe is not None
        if self._binarizer is None:
            self._binarizer = Binarize(
                onset=0.5, offset=0.5, min_duration_on=0.0, min_duration_off=0.0
            )

        # The library uses ch0 only (single distant microphone). Mirror that.
        # AudioBundle.waveform is already mono float32, so shape becomes (1, N).
        wav_t = torch.as_tensor(waveform, dtype=torch.float32).unsqueeze(0)
        sample = {"waveform": wav_t, "sample_rate": sample_rate}

        segmentations = pipe.get_segmentations(sample, soft=False)
        if pipe.apply_median_filtering:
            from scipy.ndimage import median_filter

            segmentations.data = median_filter(
                segmentations.data, size=(1, 11, 1), mode="reflect"
            )
        binarized = segmentations  # powerset → already discrete

        count = pipe.speaker_count(
            binarized,
            # DiariZen/pyannote private API: kept in sync with upstream __call__.
            pipe._segmentation.model._receptive_field,
            warm_up=(0.0, 0.0),
        )
        embeddings = pipe.get_embeddings(
            sample,
            binarized,
            exclude_overlap=pipe.embedding_exclude_overlap,
        )
        hard_clusters, _, _ = pipe.clustering(
            embeddings=embeddings,
            segmentations=binarized,
            min_clusters=pipe.min_speakers,
            max_clusters=pipe.max_speakers,
        )
        # Cap by max_speakers — segmentation can overcount instantaneously.
        count.data = np.minimum(count.data, pipe.max_speakers).astype(np.int8)

        inactive = ~np.any(binarized.data, axis=1)
        hard_clusters[inactive] = -2

        # Edge case: when no live cluster survives the inactive-mask wipe,
        # ``hard_clusters.max() == -2`` and pyannote's reconstruct() does
        # ``np.zeros((n_chunks, n_frames, hard_clusters.max() + 1))`` →
        # ValueError("negative dimensions are not allowed"). This is a
        # pre-existing DiariZen / pyannote 3.x bug, reproducible against the
        # file-based __call__ too. Treat it as "no speakers detected" and
        # return an empty Annotation; the rest of the pipeline already
        # handles empty diarization (VAD → 0 segments → empty JSON).
        if hard_clusters.size == 0 or hard_clusters.max() < 0:
            from pyannote.core import Annotation

            return Annotation()

        discrete, _ = pipe.reconstruct(segmentations, hard_clusters, count)

        return self._binarizer(discrete)

    def release(self) -> None:
        self._pipeline = None
        self._binarizer = None
