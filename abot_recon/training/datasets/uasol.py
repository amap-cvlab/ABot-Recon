# Adapted from CUT3R and the development loaders; see THIRD_PARTY_NOTICES.md.
# CUT3R-derived portions retain CC BY-NC-SA 4.0.
"""UASOL CUT3R exports: scene/{rgb,depth,cam}, already in metric units."""

import re

from ._processed_video import ProcessedVideoDataset


def _frame_number(name):
    match = re.search(r"\d+", name)
    return int(match.group()) if match else 0


class UASOL(ProcessedVideoDataset):
    dataset_name = "uasol"
    fix_interval_prob = 0.75
    depth_limit = 20.0
    depth_limit_inclusive = True

    def __init__(self, root, *args, min_interval=1, max_interval=40, **kwargs):
        super().__init__(
            root, *args, min_interval=min_interval, max_interval=max_interval, **kwargs
        )

    def _find_sequences(self):
        # UASOL has no split directory; the supplied root selects its scenes.
        return [
            (directory, self._basenames(directory, key=_frame_number))
            for directory in sorted(self.root.iterdir())
            if directory.is_dir()
        ]
