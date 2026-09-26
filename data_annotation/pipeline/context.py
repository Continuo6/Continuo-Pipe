"""Per-audio pipeline context.

Holds the mutable state that flows through the stages for a single audio
file. Each stage reads what it needs and writes its own outputs. Stages do
not read or write any module-level globals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.types import AudioBundle, DiarizationFrame, Segment


@dataclass
class PipelineContext:
    audio_path: str
    save_path: str
    audio_name: str

    audio: "AudioBundle | None" = None
    diarization: "list[DiarizationFrame] | None" = None
    segments: "list[Segment]" = field(default_factory=list)
    #: Every short segment that the VAD produced (and Scoring
    #: rated) appears here, with ``extra["kept"]`` and
    #: ``extra["dropped_reason"]`` annotating its fate. Used by the
    #: exporter to write ``<sid>.all_shorts.json`` for post-hoc debug.
    segments_all: "list[Segment]" = field(default_factory=list)
    #: Long single-speaker chunks (greedy merge of ``segments`` by
    #: :class:`pipeline.stages.long_chunk_stage.LongChunkStage`). Each
    #: long-chunk's ``extra["source_indices"]`` lists which positions in
    #: ``segments`` were merged into it. Stays empty when long-chunk
    #: emission is disabled in the config.
    long_chunks: "list[Segment]" = field(default_factory=list)
    #: All merged long chunks before the ``min_duration_s`` /
    #: ``min_mean_dnsmos`` floor in LongChunkStage. Each carries
    #: ``extra["kept"]`` and ``extra["dropped_reason"]``. Used by the
    #: exporter to write ``<sid>.all_longs.json``.
    long_chunks_all: "list[Segment]" = field(default_factory=list)


    overlap_breaks: "list[tuple[float, float]]" = field(default_factory=list)
    avg_quality: float | None = None

    final_json_path: str = ""
