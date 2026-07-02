"""
pattern_extraction/preprocessor.py
------------------------------------
Standardised text preprocessing for the LORIS framework.

Provides a configurable :class:`Preprocessor` that can normalise text before
predicate evaluation.  Supported operations:

    - Lemmatisation (via spaCy)
    - Date normalisation (ISO / US / long-form → ``<DATE>``)
    - Number/currency/percent normalisation → ``<CURRENCY>`` / ``<PERCENT>``
    - Email normalisation → ``<EMAIL>``
    - URL normalisation → ``<URL>``
    - Lowercasing

All operations are optional and disabled by default.  The preprocessor can be
applied to raw text strings or to :class:`~pattern_extraction.document.Document`
objects via :meth:`process_document`.
"""

from __future__ import annotations

import re
from copy import copy
from typing import Optional

from loris.document import Document

# ---------------------------------------------------------------------------
# Regex patterns for normalisation
# ---------------------------------------------------------------------------

# ISO dates: 2024-03-01, 2024/03/01
_RE_DATE_ISO = re.compile(
    r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'
)

# US dates: 03/01/2024, 03-01-2024
_RE_DATE_US = re.compile(
    r'\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b'
)

# Long-form dates: March 1, 2024 / January 15, 2024 / 1 March 2024
_RE_DATE_LONG = re.compile(
    r'\b(?:January|February|March|April|May|June|July|August|September|'
    r'October|November|December)\s+\d{1,2},?\s+\d{4}\b'
    r'|'
    r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+\d{4}\b',
    re.IGNORECASE,
)

# Currency: $50M, $1.5B, $100, EUR 50, etc.
_RE_CURRENCY = re.compile(
    r'(?:[\$\u20AC\u00A3])\s*\d+(?:[.,]\d+)*\s*[KMBTkmbt]?\b'
    r'|'
    r'\b\d+(?:[.,]\d+)*\s*(?:dollars?|euros?|pounds?|USD|EUR|GBP)\b',
    re.IGNORECASE,
)

# Percentages: 70%, 3.5%
_RE_PERCENT = re.compile(
    r'\b\d+(?:\.\d+)?\s*%'
)

# Email addresses
_RE_EMAIL = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
)

# URLs
_RE_URL = re.compile(
    r'https?://[^\s<>\"\']+|www\.[^\s<>\"\']+',
    re.IGNORECASE,
)


class Preprocessor:
    """
    Configurable text preprocessor for the LORIS framework.

    Parameters
    ----------
    lemmatize : bool
        If True, apply lemmatisation via spaCy.
    normalize_dates : bool
        If True, replace dates with ``<DATE>``.
    normalize_numbers : bool
        If True, replace currency amounts with ``<CURRENCY>`` and
        percentages with ``<PERCENT>``.
    normalize_emails : bool
        If True, replace email addresses with ``<EMAIL>``.
    normalize_urls : bool
        If True, replace URLs with ``<URL>``.
    lowercase : bool
        If True, convert text to lowercase (applied last).
    spacy_model : str
        spaCy model name for lemmatisation.  Default ``"en_core_web_sm"``.

    Examples
    --------
    >>> prep = Preprocessor(lemmatize=True, normalize_dates=True)
    >>> prep("The patients were diagnosed on 2024-03-01.")
    'the patient be diagnose on <DATE> .'
    """

    def __init__(
        self,
        lemmatize: bool = False,
        normalize_dates: bool = False,
        normalize_numbers: bool = False,
        normalize_emails: bool = False,
        normalize_urls: bool = False,
        lowercase: bool = False,
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        self.lemmatize = lemmatize
        self.normalize_dates = normalize_dates
        self.normalize_numbers = normalize_numbers
        self.normalize_emails = normalize_emails
        self.normalize_urls = normalize_urls
        self.lowercase = lowercase
        self.spacy_model = spacy_model
        self._nlp = None  # lazy-loaded spaCy model

    def _get_nlp(self):
        """Lazily load the spaCy model."""
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load(self.spacy_model)
        return self._nlp

    def __call__(self, text: str) -> str:
        """
        Apply all enabled preprocessing steps to *text*.

        Processing order: URLs → emails → dates → numbers → lemmatise → lowercase.
        """
        # URL normalisation (before other steps to avoid partial matches)
        if self.normalize_urls:
            text = _RE_URL.sub("<URL>", text)

        # Email normalisation
        if self.normalize_emails:
            text = _RE_EMAIL.sub("<EMAIL>", text)

        # Date normalisation
        if self.normalize_dates:
            text = _RE_DATE_LONG.sub("<DATE>", text)
            text = _RE_DATE_ISO.sub("<DATE>", text)
            text = _RE_DATE_US.sub("<DATE>", text)

        # Number normalisation (currency, then percent)
        if self.normalize_numbers:
            text = _RE_CURRENCY.sub("<CURRENCY>", text)
            text = _RE_PERCENT.sub("<PERCENT>", text)

        # Lemmatisation
        if self.lemmatize:
            nlp = self._get_nlp()
            doc = nlp(text)
            text = " ".join(token.lemma_ for token in doc)

        # Lowercasing (applied last)
        if self.lowercase:
            text = text.lower()

        return text

    def process_document(self, doc: Document) -> Document:
        """
        Apply preprocessing to all text fields of a :class:`Document`.

        Returns a **new** Document with preprocessed ``cnt``, ``ttl``, and
        ``mtd`` fields.  The ``lbl`` set is copied unchanged.

        Parameters
        ----------
        doc : Document

        Returns
        -------
        Document
            A new document with preprocessed text fields.
        """
        return Document(
            cnt=self(doc.cnt),
            ttl=self(doc.ttl) if doc.ttl else doc.ttl,
            mtd=self(doc.mtd) if doc.mtd else doc.mtd,
            lbl=set(doc.lbl),
        )

    def __repr__(self) -> str:
        flags = []
        if self.lemmatize:
            flags.append("lemmatize")
        if self.normalize_dates:
            flags.append("normalize_dates")
        if self.normalize_numbers:
            flags.append("normalize_numbers")
        if self.normalize_emails:
            flags.append("normalize_emails")
        if self.normalize_urls:
            flags.append("normalize_urls")
        if self.lowercase:
            flags.append("lowercase")
        return f"Preprocessor({', '.join(flags) if flags else 'no-op'})"
