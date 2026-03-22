#!/usr/bin/env python3
"""
MetaPAD Preprocessing Pipeline (Modern Python Replacement)
==========================================================

Converts raw text documents into MetaPAD-compatible input format using spaCy,
replacing the legacy Stanford CoreNLP + AutoPhrase dependencies.

Two output formats:
  1. XML format  — <PERSON>Barack Obama</PERSON> visited <LOCATION>Paris</LOCATION>
     (MetaPAD's native corpus.txt format)
  2. Dollar format — $PERSON visited $LOCATION
     (Entity-replaced, lowercased, phrase-chunked tokens)

Usage:
    python metapad_preprocess.py --input raw_texts/ --output data/ --format both
    python metapad_preprocess.py --input corpus.txt  --output data/ --format xml
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple

import spacy
from spacy.tokens import Doc, Span

logger = logging.getLogger(__name__)

# ── Entity-type mapping (spaCy → MetaPAD coarse types) ──────────────────────
# MetaPAD typically works with a small, fixed set of entity types.
# Map spaCy's fine-grained NER labels to coarser categories.
ENTITY_MAP = {
    # Person
    "PERSON": "PERSON",
    # Organizations
    "ORG": "ORGANIZATION",
    "NORP": "ORGANIZATION",       # nationalities / religious / political groups
    # Locations
    "GPE": "LOCATION",            # geopolitical entities (countries, cities …)
    "LOC": "LOCATION",
    "FAC": "LOCATION",            # facilities (airports, highways …)
    # Dates & times
    "DATE": "DATE",
    "TIME": "TIME",
    # Numeric / monetary
    "MONEY": "MONEY",
    "PERCENT": "PERCENT",
    "CARDINAL": "NUMBER",
    "ORDINAL": "NUMBER",
    "QUANTITY": "NUMBER",
    # Misc
    "EVENT": "EVENT",
    "WORK_OF_ART": "WORK",
    "LAW": "LAW",
    "LANGUAGE": "LANGUAGE",
    "PRODUCT": "PRODUCT",
}

# Entity types to actually tag (skip overly generic ones if desired)
ACTIVE_TYPES = {
    "PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME",
    "MONEY", "PERCENT", "NUMBER", "EVENT", "WORK", "PRODUCT",
}


# ── Helper: noun-phrase chunking ─────────────────────────────────────────────

def _merge_noun_chunks(doc: Doc) -> List[Tuple[int, int, str]]:
    """Return (start, end, underscore_text) for noun chunks that are NOT
    already covered by a named entity.  Only multi-word chunks are merged."""
    ent_spans = set()
    for ent in doc.ents:
        for i in range(ent.start, ent.end):
            ent_spans.add(i)

    chunks: List[Tuple[int, int, str]] = []
    for chunk in doc.noun_chunks:
        # Skip single-token chunks and chunks overlapping entities
        if chunk.end - chunk.start < 2:
            continue
        if any(i in ent_spans for i in range(chunk.start, chunk.end)):
            continue
        # Drop leading determiners / pronouns for cleaner phrases
        start = chunk.start
        while start < chunk.end and doc[start].pos_ in ("DET", "PRON", "ADV"):
            start += 1
        if chunk.end - start < 2:
            continue
        merged = "_".join(tok.text for tok in doc[start:chunk.end])
        chunks.append((start, chunk.end, merged))
    return chunks


# ── Core conversion functions ────────────────────────────────────────────────

def sentence_to_xml(sent: Span, phrase_chunks: List[Tuple[int, int, str]]) -> str:
    """Convert a spaCy Sentence span → MetaPAD XML line.

    Example output:
        <PERSON>Barack Obama</PERSON> visited <LOCATION>Paris</LOCATION>
    """
    tokens: List[str] = []
    i = sent.start
    # Build lookup: start_idx → (end_idx, merged_text) for noun chunks in this sentence
    chunk_map = {s: (e, t) for s, e, t in phrase_chunks if s >= sent.start and e <= sent.end}

    while i < sent.end:
        tok = sent.doc[i]

        # ── Named entity? ──
        if tok.ent_iob_ == "B":
            ent_label = ENTITY_MAP.get(tok.ent_type_, None)
            ent_end = i + 1
            while ent_end < sent.end and sent.doc[ent_end].ent_iob_ == "I":
                ent_end += 1
            ent_text = " ".join(sent.doc[j].text for j in range(i, ent_end))
            if ent_label and ent_label in ACTIVE_TYPES:
                tokens.append(f"<{ent_label}>{ent_text}</{ent_label}>")
            else:
                tokens.append(ent_text)
            i = ent_end
            continue

        # ── Noun-phrase chunk? ──
        if i in chunk_map:
            end, merged = chunk_map[i]
            tokens.append(merged)
            i = end
            continue

        # ── Regular token ──
        tokens.append(tok.text)
        i += 1

    return " ".join(tokens)


def sentence_to_dollar(sent: Span, phrase_chunks: List[Tuple[int, int, str]]) -> str:
    """Convert a spaCy Sentence span → dollar-tagged, lowercased line.

    Example output:
        $PERSON visited $LOCATION
    """
    tokens: List[str] = []
    i = sent.start
    chunk_map = {s: (e, t) for s, e, t in phrase_chunks if s >= sent.start and e <= sent.end}

    while i < sent.end:
        tok = sent.doc[i]

        # ── Named entity? ──
        if tok.ent_iob_ == "B":
            ent_label = ENTITY_MAP.get(tok.ent_type_, None)
            ent_end = i + 1
            while ent_end < sent.end and sent.doc[ent_end].ent_iob_ == "I":
                ent_end += 1
            if ent_label and ent_label in ACTIVE_TYPES:
                tokens.append(f"${ent_label}")
            else:
                ent_text = "_".join(sent.doc[j].text.lower() for j in range(i, ent_end))
                tokens.append(ent_text)
            i = ent_end
            continue

        # ── Noun-phrase chunk? ──
        if i in chunk_map:
            end, merged = chunk_map[i]
            tokens.append(merged.lower())
            i = end
            continue

        # ── Regular token → lowercase ──
        tokens.append(tok.text.lower())
        i += 1

    return " ".join(tokens)


# ── Document-level processing ────────────────────────────────────────────────

def process_text(nlp, text: str, fmt: str = "both") -> dict:
    """Process a raw text string and return converted lines.

    Args:
        nlp:  Loaded spaCy model.
        text: Raw input text (may contain multiple sentences).
        fmt:  "xml", "dollar", or "both".

    Returns:
        dict with keys "xml" and/or "dollar", each a list of strings (one per sentence).
    """
    doc = nlp(text)
    phrase_chunks = _merge_noun_chunks(doc)

    result = {}
    if fmt in ("xml", "both"):
        result["xml"] = [sentence_to_xml(s, phrase_chunks) for s in doc.sents]
    if fmt in ("dollar", "both"):
        result["dollar"] = [sentence_to_dollar(s, phrase_chunks) for s in doc.sents]
    return result


# ── I/O helpers ──────────────────────────────────────────────────────────────

def read_input(path: Path) -> List[str]:
    """Read input — either a single .txt file or all .txt files under a directory."""
    texts: List[str] = []
    if path.is_file():
        texts.append(path.read_text(encoding="utf-8"))
    elif path.is_dir():
        for f in sorted(path.rglob("*.txt")):
            texts.append(f.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError(f"Input path not found: {path}")
    if not texts:
        raise ValueError(f"No .txt files found under {path}")
    return texts


def write_output(lines: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %d lines → %s", len(lines), path)


# ── Main ─────────────────────────────────────────────────────────────────────

def build_pipeline(model_name: str = "en_core_web_sm") -> spacy.Language:
    """Load spaCy model. Falls back to download if not present."""
    try:
        nlp = spacy.load(model_name)
    except OSError:
        logger.info("Downloading spaCy model '%s' …", model_name)
        spacy.cli.download(model_name)  # type: ignore[attr-defined]
        nlp = spacy.load(model_name)
    # Increase max length for large documents
    nlp.max_length = 5_000_000
    return nlp


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw text → MetaPAD input format using spaCy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-i", "--input", type=Path, required=True,
        help="Path to a .txt file or a directory of .txt files.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("data"),
        help="Output directory (default: ./data).",
    )
    parser.add_argument(
        "-f", "--format", choices=["xml", "dollar", "both"], default="both",
        help="Output format: 'xml' (MetaPAD corpus.txt), 'dollar' ($TYPE tags), or 'both'.",
    )
    parser.add_argument(
        "-m", "--model", default="en_core_web_sm",
        help="spaCy model name (default: en_core_web_sm). Use en_core_web_trf for best accuracy.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="spaCy nlp.pipe batch size (default: 256).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("Loading spaCy model: %s", args.model)
    nlp = build_pipeline(args.model)

    logger.info("Reading input from: %s", args.input)
    texts = read_input(args.input)
    logger.info("Read %d document(s)", len(texts))

    xml_lines: List[str] = []
    dollar_lines: List[str] = []

    # Process in batches via nlp.pipe for efficiency
    for doc in nlp.pipe(texts, batch_size=args.batch_size):
        phrase_chunks = _merge_noun_chunks(doc)
        for sent in doc.sents:
            if args.format in ("xml", "both"):
                xml_lines.append(sentence_to_xml(sent, phrase_chunks))
            if args.format in ("dollar", "both"):
                dollar_lines.append(sentence_to_dollar(sent, phrase_chunks))

    # Write outputs
    out = args.output
    if xml_lines:
        write_output(xml_lines, out / "corpus_xml.txt")
    if dollar_lines:
        write_output(dollar_lines, out / "corpus_dollar.txt")

    logger.info("Done. Processed %d sentences total.",
                max(len(xml_lines), len(dollar_lines)))


if __name__ == "__main__":
    main()
