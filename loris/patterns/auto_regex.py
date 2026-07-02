"""
pattern_extraction/auto_regex.py
---------------------------------
Automatic high-frequency structural pattern extractor.

Scans a corpus for tokens with structural shapes (mixed alpha-numeric,
separators, version-like patterns, etc.), groups them by a generalised
"shape signature", and converts high-frequency shapes into regex patterns.

These auto-discovered regexes serve as additional anchor sources for the
LORIS candidate retrieval step (Step 2), complementing the static regex
patterns loaded from ``regex_patterns.yaml``.

Algorithm
---------
1. **Tokenize** each document by whitespace and punctuation boundaries.
2. **Filter** to structural tokens (mixed alpha+digit, separators, etc.).
3. **Shape signature**: generalise each token (``A`` for uppercase letter,
   ``a`` for lowercase, ``0`` for digit, separators preserved).
4. **Frequency count** across the entire corpus.
5. **Shape -> regex**: convert shapes exceeding ``min_freq`` to regex.
6. **Dedup** against any existing static regexes.

Usage
-----
::

    extractor = AutoRegexExtractor(min_freq=3)
    extractor.fit(corpus_texts)
    patterns = extractor.get_patterns()          # [(name, compiled_re), ...]
    anchors  = extractor.get_anchors(corpus_texts)  # [str, ...]
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Tokenization regex: split on whitespace and common punctuation boundaries,
# but keep tokens that contain internal separators like - _ . / :
_TOKEN_RE = re.compile(r"[^\s,;!?\"'()\[\]{}<>]+")


class AutoRegexExtractor:
    """
    Discover high-frequency structural token patterns from a corpus.

    Parameters
    ----------
    min_freq : int
        Minimum number of occurrences (across the entire corpus) for a shape
        to be converted into a regex pattern.  Default 3.
    min_token_len : int
        Tokens shorter than this are ignored.  Default 3.
    max_token_len : int
        Tokens longer than this are ignored (likely URLs or noise).  Default 40.
    """

    def __init__(
        self,
        min_freq: int = 3,
        min_token_len: int = 3,
        max_token_len: int = 40,
    ) -> None:
        self.min_freq = min_freq
        self.min_token_len = min_token_len
        self.max_token_len = max_token_len

        # Populated by fit()
        self._shape_counts: Counter = Counter()
        self._shape_examples: Dict[str, str] = {}  # shape -> first example
        self._patterns: List[Tuple[str, re.Pattern[str]]] = []
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        texts: List[str],
        existing_regexes: Optional[List[Tuple[str, re.Pattern[str]]]] = None,
    ) -> "AutoRegexExtractor":
        """
        Scan *texts* to discover high-frequency structural patterns.

        Parameters
        ----------
        texts : List[str]
            Corpus documents.
        existing_regexes : list of (name, compiled_re), optional
            Static regexes already loaded (from ``regex_patterns.yaml``).
            Discovered patterns that overlap with these are removed.

        Returns
        -------
        AutoRegexExtractor
            ``self``, for method chaining.
        """
        self._shape_counts.clear()
        self._shape_examples.clear()

        # 1-3. Tokenize, filter, compute shapes
        for text in texts:
            tokens = self._tokenize(text)
            for tok in tokens:
                if not self._is_structural(tok):
                    continue
                shape = self._shape_signature(tok)
                self._shape_counts[shape] += 1
                if shape not in self._shape_examples:
                    self._shape_examples[shape] = tok

        # 4-5. Convert high-frequency shapes to regex
        raw_patterns: List[Tuple[str, str, re.Pattern[str]]] = []
        for shape, count in self._shape_counts.items():
            if count >= self.min_freq:
                regex_str = self._shape_to_regex(shape)
                name = f"auto:{self._shape_examples.get(shape, shape)}"
                try:
                    compiled = re.compile(regex_str)
                    raw_patterns.append((name, regex_str, compiled))
                except re.error:
                    logger.debug(
                        "Skipping invalid auto-regex for shape %r: %s",
                        shape, regex_str,
                    )

        # 6. Dedup against existing static regexes
        if existing_regexes:
            raw_patterns = self._dedup_against(raw_patterns, existing_regexes)

        self._patterns = [(name, comp) for name, _, comp in raw_patterns]
        self._is_fitted = True

        logger.info(
            "AutoRegexExtractor.fit(): %d shapes seen, %d above min_freq=%d, "
            "%d patterns after dedup.",
            len(self._shape_counts),
            sum(1 for c in self._shape_counts.values() if c >= self.min_freq),
            self.min_freq,
            len(self._patterns),
        )
        return self

    def get_patterns(self) -> List[Tuple[str, re.Pattern[str]]]:
        """
        Return discovered patterns as ``(name, compiled_regex)`` tuples.

        Format is compatible with :func:`load_regex_config` output, so the
        patterns can be appended directly to ``self._entity_regexes`` in
        :class:`PatternAbstractor`.
        """
        return list(self._patterns)

    def get_anchors(self, texts: List[str]) -> List[str]:
        """
        Apply discovered patterns to *texts* and return matched spans.

        Returns
        -------
        List[str]
            Unique matched strings (lowercased), suitable for direct use
            as anchors in ``_collect_anchors``.
        """
        if not self._patterns:
            return []

        anchors: set = set()
        for text in texts:
            for _, pattern in self._patterns:
                for m in pattern.finditer(text):
                    span = m.group(0).lower().strip()
                    if span:
                        anchors.add(span)
        return list(anchors)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Split text into tokens, keeping internal separators."""
        tokens = []
        for tok in _TOKEN_RE.findall(text):
            # Strip trailing punctuation that isn't a structural separator
            tok = tok.rstrip(".")
            if self.min_token_len <= len(tok) <= self.max_token_len:
                tokens.append(tok)
        return tokens

    def _is_structural(self, token: str) -> bool:
        """
        Return True if *token* has structural characteristics.

        A token is structural if it contains:
        - Mixed alpha + digit characters, OR
        - Internal separators (``-``, ``_``, ``.``, ``/``, ``:``) combined
          with at least one alphanumeric, OR
        - All-uppercase letters followed/preceded by digits (e.g. ``ERR4501``)
        """
        has_alpha = False
        has_digit = False
        has_sep = False

        for ch in token:
            if ch.isalpha():
                has_alpha = True
            elif ch.isdigit():
                has_digit = True
            elif ch in "-_./:" :
                has_sep = True

        # Mixed alpha+digit (e.g. ERR4501, v2, AB12)
        if has_alpha and has_digit:
            return True

        # Separator + at least some alphanumeric content on both sides
        # e.g. "2024-03-01", "ORD-001"
        if has_sep and (has_alpha or has_digit):
            parts = re.split(r"[-_./:]", token)
            non_empty = [p for p in parts if p]
            if len(non_empty) >= 2:
                return True

        return False

    def _shape_signature(self, token: str) -> str:
        """
        Generalise *token* to a shape signature.

        Rules:
        - Uppercase letter -> ``A``
        - Lowercase letter -> ``a``
        - Digit -> ``0``
        - Characters in ``-_./:``: preserved as-is
        - Everything else -> ``?``

        Examples::

            ERR-4501       -> AAA-0000
            v2.1.0         -> a0.0.0
            2024-03-01     -> 0000-00-00
            ORD-20240301-AB12 -> AAA-00000000-AA00
        """
        chars = []
        for ch in token:
            if ch.isupper():
                chars.append("A")
            elif ch.islower():
                chars.append("a")
            elif ch.isdigit():
                chars.append("0")
            elif ch in "-_./:" :
                chars.append(ch)
            else:
                chars.append("?")
        return "".join(chars)

    def _shape_to_regex(self, shape: str) -> str:
        r"""
        Convert a shape signature to a regex pattern string.

        Consecutive identical shape chars are collapsed into quantified
        character classes:

        - ``AAA`` -> ``[A-Z]{3}``
        - ``aaa`` -> ``[a-z]{3}``
        - ``000`` -> ``\d{3}``
        - Separators are escaped literally
        - ``?`` segments become ``.``

        Word boundaries ``\b`` are prepended and appended.
        """
        parts: List[str] = []
        i = 0
        while i < len(shape):
            ch = shape[i]
            # Count consecutive identical characters
            j = i + 1
            while j < len(shape) and shape[j] == ch:
                j += 1
            run_len = j - i

            if ch == "A":
                parts.append(f"[A-Z]{{{run_len}}}")
            elif ch == "a":
                parts.append(f"[a-z]{{{run_len}}}")
            elif ch == "0":
                parts.append(rf"\d{{{run_len}}}")
            elif ch in "-_./:" :
                parts.append(re.escape(ch) * run_len)
            else:
                parts.append("." * run_len)

            i = j

        regex_body = "".join(parts)
        return rf"\b{regex_body}\b"

    def _dedup_against(
        self,
        candidates: List[Tuple[str, str, re.Pattern[str]]],
        existing: List[Tuple[str, re.Pattern[str]]],
    ) -> List[Tuple[str, str, re.Pattern[str]]]:
        """
        Remove candidate patterns whose regex string duplicates an existing
        static pattern.

        Comparison is on the pattern string (``pattern.pattern``) to catch
        exact duplicates.  We also check if the candidate's example token is
        already matched by any existing regex (semantic overlap).
        """
        existing_strs = {pat.pattern for _, pat in existing}

        result = []
        for name, regex_str, compiled in candidates:
            # Skip if the regex string is already in static patterns
            if regex_str in existing_strs:
                continue

            # Check if the example token this shape was derived from is
            # already matched by an existing regex
            shape_example = name.replace("auto:", "", 1)
            already_covered = False
            for _, ex_pat in existing:
                if ex_pat.search(shape_example):
                    already_covered = True
                    break

            if already_covered:
                logger.debug(
                    "Auto-regex %r overlaps with existing static regex; skipping.",
                    name,
                )
                continue

            result.append((name, regex_str, compiled))

        return result

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        n_pat = len(self._patterns) if self._is_fitted else "?"
        return (
            f"AutoRegexExtractor(min_freq={self.min_freq}, "
            f"patterns={n_pat}, status={status})"
        )
