"""
serialization.py — shard writers and readers for SentenceTokenization records.

Supports JSONL and Parquet output formats.
Each record is a flat dict representation of SentenceTokenization with
nested morph analyses encoded as dicts (JSON-serializable).
"""
from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

from sarftok import MorphAnalysis, SentenceTokenization, WordMorphBundle

# ---------------------------------------------------------------------------
# Data model → dict / dict → data model
# ---------------------------------------------------------------------------


def analysis_to_dict(a: MorphAnalysis) -> dict:
    return {
        "prob": a.prob,
        "proclitics": a.proclitics,
        "root": a.root,
        "pattern": a.pattern,
        "enclitics": a.enclitics,
        "pos": a.pos,
        "lemma": a.lemma,
        "is_oov_root": a.is_oov_root,
        "is_oov_pattern": a.is_oov_pattern,
        "source": a.source,
    }


def dict_to_analysis(d: dict) -> MorphAnalysis:
    return MorphAnalysis(
        prob=d.get("prob", 0.0),
        proclitics=d.get("proclitics", []),
        root=d.get("root"),
        pattern=d.get("pattern"),
        enclitics=d.get("enclitics", []),
        pos=d.get("pos"),
        lemma=d.get("lemma"),
        is_oov_root=d.get("is_oov_root", False),
        is_oov_pattern=d.get("is_oov_pattern", False),
        source=d.get("source", "unknown"),
    )


def word_bundle_to_dict(w: WordMorphBundle) -> dict:
    return {
        "raw_word": w.raw_word,
        "normalized_word": w.normalized_word,
        "analyses": [analysis_to_dict(a) for a in w.analyses],
        "surface_pieces": w.surface_pieces,
        "word_index": w.word_index,
    }


def dict_to_word_bundle(d: dict) -> WordMorphBundle:
    return WordMorphBundle(
        raw_word=d.get("raw_word", ""),
        normalized_word=d.get("normalized_word", ""),
        analyses=[dict_to_analysis(a) for a in d.get("analyses", [])],
        surface_pieces=d.get("surface_pieces", []),
        word_index=d.get("word_index", 0),
    )


def sentence_to_dict(s: SentenceTokenization) -> dict:
    return {
        "raw_text": s.raw_text,
        "normalized_text": s.normalized_text,
        "words": [word_bundle_to_dict(w) for w in s.words],
        "surface_input_ids": s.surface_input_ids,
        "word_to_surface_spans": [list(sp) for sp in s.word_to_surface_spans],
    }


def dict_to_sentence(d: dict) -> SentenceTokenization:
    return SentenceTokenization(
        raw_text=d.get("raw_text", ""),
        normalized_text=d.get("normalized_text", ""),
        words=[dict_to_word_bundle(w) for w in d.get("words", [])],
        surface_input_ids=d.get("surface_input_ids", []),
        word_to_surface_spans=[tuple(sp) for sp in d.get("word_to_surface_spans", [])],
    )


# ---------------------------------------------------------------------------
# JSONL writer / reader
# ---------------------------------------------------------------------------


class JsonlShardWriter:
    """Write SentenceTokenization records to JSONL shards.

    Parameters
    ----------
    output_dir:
        Directory to write shards to.
    shard_size:
        Number of records per shard file.
    prefix:
        Filename prefix for shards.
    """

    def __init__(
        self,
        output_dir: str | Path,
        shard_size: int = 10_000,
        prefix: str = "shard",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.prefix = prefix
        self._current_shard: int = 0
        self._current_count: int = 0
        self._file = None
        self._open_shard()

    def _open_shard(self) -> None:
        if self._file:
            self._file.close()
        path = self.output_dir / f"{self.prefix}_{self._current_shard:05d}.jsonl"
        self._file = open(path, "w", encoding="utf-8")

    def write(self, sentence: SentenceTokenization) -> None:
        """Write one sentence tokenization record."""
        line = json.dumps(sentence_to_dict(sentence), ensure_ascii=False)
        self._file.write(line + "\n")
        self._current_count += 1
        if self._current_count >= self.shard_size:
            self._current_shard += 1
            self._current_count = 0
            self._open_shard()

    def write_batch(self, sentences: list[SentenceTokenization]) -> None:
        for s in sentences:
            self.write(s)

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self) -> JsonlShardWriter:
        return self

    def __exit__(self, *args) -> None:
        self.close()


def read_jsonl_shards(
    shard_dir: str | Path,
    pattern: str = "*.jsonl",
) -> Generator[SentenceTokenization, None, None]:
    """Lazily iterate over all JSONL shards in *shard_dir*."""
    shard_dir = Path(shard_dir)
    for path in sorted(shard_dir.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield dict_to_sentence(json.loads(line))


# ---------------------------------------------------------------------------
# Parquet writer / reader (optional — requires pyarrow)
# ---------------------------------------------------------------------------


def write_parquet_shard(
    sentences: list[SentenceTokenization],
    path: str | Path,
) -> None:
    """Write a list of SentenceTokenization records to a Parquet file."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required for Parquet output: pip install pyarrow")

    rows = [sentence_to_dict(s) for s in sentences]
    # Flatten nested structures to JSON strings for Parquet compatibility
    flat_rows = []
    for r in rows:
        flat_rows.append({
            "raw_text": r["raw_text"],
            "normalized_text": r["normalized_text"],
            "words_json": json.dumps(r["words"], ensure_ascii=False),
            "surface_input_ids": r["surface_input_ids"],
            "word_to_surface_spans_json": json.dumps(
                r["word_to_surface_spans"], ensure_ascii=False
            ),
        })

    table = pa.Table.from_pylist(flat_rows)
    pq.write_table(table, str(path))


def read_parquet_shards(
    shard_dir: str | Path,
    pattern: str = "*.parquet",
) -> Generator[SentenceTokenization, None, None]:
    """Lazily iterate over all Parquet shards in *shard_dir*."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required for Parquet reading: pip install pyarrow")

    shard_dir = Path(shard_dir)
    for path in sorted(shard_dir.glob(pattern)):
        table = pq.read_table(str(path))
        for batch in table.to_batches():
            batch_dict = batch.to_pydict()
            n = len(batch_dict["raw_text"])
            for i in range(n):
                d = {
                    "raw_text": batch_dict["raw_text"][i],
                    "normalized_text": batch_dict["normalized_text"][i],
                    "words": json.loads(batch_dict["words_json"][i]),
                    "surface_input_ids": batch_dict["surface_input_ids"][i],
                    "word_to_surface_spans": json.loads(
                        batch_dict["word_to_surface_spans_json"][i]
                    ),
                }
                yield dict_to_sentence(d)
