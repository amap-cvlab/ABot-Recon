# Adapted from CUT3R and the development loaders; see THIRD_PARTY_NOTICES.md.
# CUT3R-derived portions retain CC BY-NC-SA 4.0.
"""PointOdyssey in the official CUT3R split/scene/{rgb,depth,cam} layout."""

from ._processed_video import ProcessedVideoDataset


class PointOdyssey(ProcessedVideoDataset):
    """Keep the source scene whitelist and metric depth (invalid above 1000 m)."""

    dataset_name = "pointodyssey"
    image_suffix = ".jpg"
    depth_limit = 1000.0
    SCENES_TO_USE = frozenset({
        "cnb_dlab_0215_3rd", "cnb_dlab_0215_ego1",
        "cnb_dlab_0225_3rd", "cnb_dlab_0225_ego1", "dancing", "dancingroom0_3rd",
        "footlab_3rd", "footlab_ego1", "footlab_ego2", "girl", "girl_egocentric",
        "human_egocentric", "human_in_scene", "human_in_scene1", "kg", "kg_ego1",
        "kg_ego2", "kitchen_gfloor", "kitchen_gfloor_ego1", "kitchen_gfloor_ego2",
        "scene_carb_h_tables", "scene_carb_h_tables_ego1", "scene_carb_h_tables_ego2",
        "scene_j716_3rd", "scene_j716_ego1", "scene_j716_ego2",
        "scene_recording_20210910_S05_S06_0_3rd", "scene_recording_20210910_S05_S06_0_ego2",
        "scene1_0129", "scene1_0129_ego", "seminar_h52_3rd", "seminar_h52_ego1",
        "seminar_h52_ego2",
    })

    def __init__(self, root, *args, min_interval=1, max_interval=4, **kwargs):
        super().__init__(
            root, *args, min_interval=min_interval, max_interval=max_interval, **kwargs
        )

    def _find_sequences(self):
        if self.split not in {"train", "test", "val"}:
            raise ValueError("PointOdyssey split must be train, test or val")
        return [
            (directory, self._basenames(directory))
            for directory in sorted((self.root / self.split).iterdir())
            if directory.is_dir() and directory.name in self.SCENES_TO_USE
        ]
