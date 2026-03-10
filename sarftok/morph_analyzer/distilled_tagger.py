"""
DistilledMorphTagger — stub for the Phase-4 distilled morphology backend.

This is a placeholder that raises a clear error until a trained model
is available.  The class interface mirrors the other backends so that
switching in production requires only a config change.
"""
from __future__ import annotations

from typing import List

from sarftok import MorphAnalysis
from sarftok.morph_analyzer.interface import MorphAnalyzer


class DistilledMorphTagger(MorphAnalyzer):
    """Contextual distilled morphology tagger (Phase 4 — not yet implemented).

    Replace this stub by implementing :meth:`_raw_analyze_word` with a
    fine-tuned sequence-labelling model that outputs top-k analyses with
    calibrated probabilities.
    """

    def __init__(self, model_path: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_path = model_path
        raise NotImplementedError(
            "DistilledMorphTagger is a Phase-4 feature and has not been "
            "implemented yet.  Use analyzer_backend='heuristic' or "
            "analyzer_backend='camel' instead."
        )

    def _raw_analyze_word(self, word: str) -> List[MorphAnalysis]:
        raise NotImplementedError("DistilledMorphTagger not implemented.")
