# Adapted from CUT3R and the development loaders; see THIRD_PARTY_NOTICES.md.
# CUT3R-derived portions retain CC BY-NC-SA 4.0.
"""DynamicReplica CUT3R exports; use the left stream as in the source loader."""

from ._processed_video import ProcessedVideoDataset


class DynamicReplica(ProcessedVideoDataset):
    dataset_name = "dynamic_replica"

    def __init__(self, root, *args, min_interval=1, max_interval=16, **kwargs):
        super().__init__(
            root, *args, min_interval=min_interval, max_interval=max_interval, **kwargs
        )

    def _find_sequences(self):
        return [
            (directory / "left", self._basenames(directory / "left", key=float))
            for directory in sorted((self.root / self.split).iterdir())
            if directory.is_dir()
        ]

    def _scene_label(self, directory):
        return directory.parent.name
