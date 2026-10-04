"""Safe regex necessary-condition indexing; exact finditer remains the oracle.

AST conjunction/disjunction is over APPROXIMATE SUPERSETS of matching rows.
Optional constructs, negative assertions and unknown operations impose no
restriction. Positive repetitions/assertions can supply necessary literals.
No sampled/learned model, document frequency cut or match approximation is
used. This is an execution optimization and never defines pattern semantics.
"""
from functools import lru_cache
import re
import sre_parse


class RegexSupersetIndex:
    def __init__(self, texts):
        self.texts = tuple(texts)
        self.all_rows = (1 << len(self.texts)) - 1
        self.special = 0
        self.grams = {}
        for row, text in enumerate(self.texts):
            bit = 1 << row
            if any(ch in text for ch in '\u0130\u0131\u017f\u212a'):
                self.special |= bit
            seen = set()
            for run in re.findall(r'[\x00-\x7f]+', text.lower()):
                seen.update(run[i:i + 3] for i in range(len(run) - 2))
                seen.update(run[i:i + 2] for i in range(len(run) - 1))
            for gram in seen:
                self.grams[gram] = self.grams.get(gram, 0) | bit

    def _literal(self, text):
        # Every match of a mandatory literal contains every one of its ASCII
        # 2/3-grams, except Python IGNORECASE's four special Unicode letters.
        # Add those rows at the FINAL result regardless of scoped flags.
        result = self.all_rows
        for run in re.findall(r'[\x00-\x7f]+', text.lower()):
            width = min(3, len(run))
            if width < 2:
                continue
            for start in range(len(run) - width + 1):
                result &= self.grams.get(run[start:start + width], 0)
        return result

    def _sequence(self, sequence):
        result, literals = self.all_rows, []

        def flush():
            if not literals:
                return self.all_rows
            text = ''.join(literals)
            literals.clear()
            return self._literal(text)

        for op, arg in sequence:
            name = str(op)
            if name == 'LITERAL':
                literals.append(chr(arg))
                continue
            result &= flush()
            if name == 'SUBPATTERN':
                result &= self._sequence(arg[-1])
            elif name == 'BRANCH':
                union = 0
                for branch in arg[1]:
                    union |= self._sequence(branch)
                result &= union
            elif name in ('MAX_REPEAT', 'MIN_REPEAT', 'POSSESSIVE_REPEAT'):
                if arg[0] > 0:
                    result &= self._sequence(arg[2])
            elif name == 'ASSERT':
                # Either lookahead or lookbehind still requires this literal
                # somewhere in the same full document, not at a guessed offset.
                result &= self._sequence(arg[1])
            elif name == 'ATOMIC_GROUP':
                result &= self._sequence(arg)
            # IN, ANY, NOT_LITERAL, GROUPREF, negative assertions and unknown
            # constructs conservatively impose no necessary text condition.
        return result & flush()

    def candidate_rows(self, raw, flags, groups=None):
        # Parsing and matching use the same Python interpreter and flags. A
        # rejected/unsupported parser construct uses the full corpus; matching
        # itself still raises its original compilation errors in the caller.
        try:
            necessary = self._sequence(sre_parse.parse(raw, flags)) | self.special
        except (ValueError, TypeError, re.error, OverflowError):
            return range(len(self.texts))
        rows = []
        while necessary:
            bit = necessary & -necessary
            rows.append(bit.bit_length() - 1)
            necessary ^= bit
        return rows
