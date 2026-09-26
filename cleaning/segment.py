"""Word segmentation for website-style names (``tristateguild.com`` -> ``tri state guild``).

Unigram Viterbi over a vocabulary of Source-1 name tokens. A string is only
segmented when every piece is a vocabulary token; otherwise ``None`` is
returned and the caller keeps the original string.
"""

from __future__ import annotations

import math


class Segmenter:
    def __init__(self, vocab: dict[str, int], max_token_len: int = 20):
        total = float(sum(vocab.values())) or 1.0
        self.logp = {t: math.log(c / total) for t, c in vocab.items() if t}
        self.maxlen = min(max_token_len, max((len(t) for t in self.logp), default=1))
        # single letters are legal only as a last resort: penalise heavily
        self.single_penalty = -12.0

    def segment(self, s: str) -> list[str] | None:
        s = s.lower()
        n = len(s)
        if n == 0 or n > 60 or not s.isalnum():
            return None
        # a string that is itself a known name token stays whole ("tristate")
        if n >= 4 and s in self.logp:
            return [s]
        best = [-math.inf] * (n + 1)
        back = [-1] * (n + 1)
        best[0] = 0.0
        for i in range(1, n + 1):
            for j in range(max(0, i - self.maxlen), i):
                if best[j] == -math.inf:
                    continue
                piece = s[j:i]
                lp = self.logp.get(piece)
                if lp is None:
                    continue
                lp -= 4.0  # per-piece penalty: prefer fewer, longer pieces
                if len(piece) == 1:
                    lp += self.single_penalty
                elif len(piece) == 2:
                    lp -= 3.0
                score = best[j] + lp
                if score > best[i]:
                    best[i] = score
                    back[i] = j
        if best[n] == -math.inf:
            return None
        out = []
        i = n
        while i > 0:
            j = back[i]
            out.append(s[j:i])
            i = j
        out.reverse()
        # refuse segmentations that are mostly single letters
        if sum(len(t) == 1 for t in out) > max(1, len(out) // 3):
            return None
        return out
