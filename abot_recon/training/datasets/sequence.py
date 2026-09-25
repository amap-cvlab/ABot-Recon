"""Prefix-sum, in-memory indexing of eligible starts in processed sequences."""

from __future__ import annotations

from bisect import bisect_right
from itertools import accumulate
from operator import index


class SequenceIndex:
    """Apply the development loaders' minimum-tail rule without a disk cache.

    ``allow_repeat`` lowers the minimum sequence/tail length to
    ``max(num_views // 3, 3)``; it does not select the sampling algorithm.
    Paths are retained as supplied and frame basenames are copied into lists.
    """

    def __init__(self, sequences, num_views, allow_repeat):
        num_views = int(num_views)
        if num_views < 1:
            raise ValueError("num_views must be positive")
        cutoff = max(num_views // 3, 3) if allow_repeat else num_views
        self.sequences = [
            (path, list(names)) for path, names in sequences if len(names) >= cutoff
        ]
        self.image_count = sum(len(names) for _, names in self.sequences)
        self._ends = list(accumulate(len(names) - cutoff + 1 for _, names in self.sequences))

    def __len__(self):
        return self._ends[-1] if self._ends else 0

    def resolve(self, item):
        item = index(item)
        if item < 0:
            item += len(self)
        if not 0 <= item < len(self):
            raise IndexError("sequence index out of range")
        sequence_id = bisect_right(self._ends, item)
        offset = self._ends[sequence_id - 1] if sequence_id else 0
        path, names = self.sequences[sequence_id]
        return path, names, item - offset
