# Adapted from CUT3R and the development loaders; see THIRD_PARTY_NOTICES.md.
# CUT3R-derived portions retain CC BY-NC-SA 4.0.
"""Processed HyperSim camera folders pooled into scene-level pose graphs."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .base import MultiViewDataset
from .io import image_open, np_load
from .unreal4k_seq import UnrealStereo4KSeq, _rotation_angle_deg


_SEQUENCE_BLACKLIST_FILENAME = "hypersim_geometry_blacklist_20260810.txt"
# Verified against both development recipes and their dataset-root override.
_DEFAULT_EXCLUDED_SEQUENCES = frozenset({
    "ai_003_001/cam_00",
    "ai_004_009/cam_01",
    "ai_031_004/cam_00",
    "ai_052_002/cam_01",
})


class HyperSimSeq(MultiViewDataset):
    """Read scene/cam_*/NNNNNN_{rgb.png,depth.npy,cam.npz} exports.

    CUT3R already converts radial range into metric camera-z depth and poses
    into OpenCV-axis camera-to-world. Do not convert either again.
    Camera folders share a scene coordinate system, but the development
    minimum-tail cutoff applies to each folder before they are pooled.
    The split argument is metadata, not a held-out scene partition.

    ``sequence_blacklist_path=None`` or ``"auto"`` uses the dated blacklist
    file at the dataset root when present, otherwise the four inline defaults.
    An explicit local text-file path replaces those defaults; relative paths
    are resolved against the dataset root. ``""`` disables the blacklist.
    Entries are scene/cam_* identifiers; blank lines and # comments are ignored.
    No generated on-disk graph cache is used.
    """

    dataset_name = "hypersim_seq"

    def __init__(
        self, root, *, split="train", allow_repeat=True,
        sequence_blacklist_path=None, max_rotation_deg=20.0,
        max_translation_factor=5.0, pose_knn=48, graph_neighbors=24,
        beam_width=192, min_unique_ratio=0.50, min_component_views=32,
        pose_load_workers=1, sequence_start_retries=6,
        sequence_scene_retries=64, **kwargs,
    ):
        self.max_rotation_deg = float(max_rotation_deg)
        self.max_translation_factor = float(max_translation_factor)
        if not np.isfinite(self.max_rotation_deg) or self.max_rotation_deg <= 0:
            raise ValueError("max_rotation_deg must be positive and finite")
        if not np.isfinite(self.max_translation_factor) or self.max_translation_factor <= 0:
            raise ValueError("max_translation_factor must be positive and finite")
        self.pose_knn = max(2, int(pose_knn))
        self.graph_neighbors = max(2, int(graph_neighbors))
        self.beam_width = max(1, int(beam_width))
        self.min_unique_ratio = float(min_unique_ratio)
        if not 0 < self.min_unique_ratio <= 1:
            raise ValueError("min_unique_ratio must be in (0, 1]")
        self.min_component_views = max(2, int(min_component_views))
        self.pose_load_workers = max(1, int(pose_load_workers))
        self.sequence_start_retries = max(1, int(sequence_start_retries))
        self.sequence_scene_retries = max(1, int(sequence_scene_retries))
        self._scene_graph_cache = {}
        self._runtime_failed_groups = set()
        super().__init__(root=root, split=split, allow_repeat=allow_repeat, **kwargs)
        self.split = split
        if self.num_views < 1:
            raise ValueError("num_views must be positive")
        self.excluded_sequences = set()
        if sequence_blacklist_path is None or sequence_blacklist_path == "auto":
            path = self.root / _SEQUENCE_BLACKLIST_FILENAME
            if not path.is_file():
                self.excluded_sequences.update(_DEFAULT_EXCLUDED_SEQUENCES)
                path = None
        elif sequence_blacklist_path == "":
            path = None
        else:
            path = Path(sequence_blacklist_path).expanduser()
            if not path.is_absolute():
                path = self.root / path
        if path is not None:
            with path.open(encoding="utf-8") as handle:
                self.excluded_sequences = {
                    line.strip() for line in handle
                    if line.strip() and not line.lstrip().startswith("#")
                }
        self._load_data()
        self._build_scene_groups()

    def _load_data(self):
        self.scenes, self.sceneids, self.images = [], [], []
        self.start_img_ids, self.scene_img_list = [], []
        cutoff = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
        for scene in sorted(path for path in self.root.iterdir() if path.is_dir()):
            for directory in sorted(scene.iterdir()):
                if not directory.is_dir() or not directory.name.startswith("cam_"):
                    continue
                subscene = directory.relative_to(self.root).as_posix()
                if subscene in self.excluded_sequences:
                    continue
                names = sorted(path.name for path in directory.glob("*_rgb.png") if path.is_file())
                if len(names) < cutoff:
                    continue
                ids = list(range(len(self.images), len(self.images) + len(names)))
                self.sceneids.extend([len(self.scenes)] * len(names))
                self.scenes.append(subscene)
                self.images.extend(names)
                self.scene_img_list.append(ids)
                self.start_img_ids.extend(ids[:len(names) - cutoff + 1])

    def __len__(self):
        return 10 * len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _build_scene_groups(self):
        groups = defaultdict(list)
        for scene_id, subscene in enumerate(self.scenes):
            groups[subscene.split("/", 1)[0]].append(scene_id)
        self._group_scene_ids = {key: tuple(value) for key, value in groups.items()}
        self._eligible_groups = tuple(sorted(groups))
        self._scene_id_to_group = {
            scene_id: key for key, ids in self._group_scene_ids.items() for scene_id in ids
        }

    @staticmethod
    def _read_pose(path):
        with np_load(path) as camera:
            pose = np.asarray(camera["pose"], dtype=np.float32).copy()
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"Invalid HyperSim pose: {path}")
        return pose

    def _group_global_ids(self, group):
        return np.asarray(
            [index for scene_id in self._group_scene_ids[group]
             for index in self.scene_img_list[scene_id]], dtype=np.int64,
        )

    def _camera_path(self, global_id):
        global_id = int(global_id)
        directory = self.root / self.scenes[self.sceneids[global_id]]
        return directory / self.images[global_id].replace("rgb.png", "cam.npz")

    def _build_group_graph(self, group: str):
        if group in self._scene_graph_cache:
            return self._scene_graph_cache[group]

        global_ids = self._group_global_ids(group)
        paths = [self._camera_path(int(global_id)) for global_id in global_ids]
        if self.pose_load_workers > 1:
            with ThreadPoolExecutor(max_workers=self.pose_load_workers) as pool:
                poses = np.stack(list(pool.map(self._read_pose, paths)))
        else:
            poses = np.stack([self._read_pose(path) for path in paths])
        rotations = poses[:, :3, :3]
        centers = poses[:, :3, 3]
        count = len(global_ids)
        if count < 2:
            self._scene_graph_cache[group] = None
            return None

        query_k = min(self.pose_knn + 1, count)
        distances, indices = cKDTree(centers).query(centers, k=query_k)
        if query_k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        nearest = distances[:, 1]
        positive = nearest[nearest > 1e-8]
        translation_scale = float(np.median(positive)) if len(positive) else 1.0
        translation_limit = self.max_translation_factor * translation_scale

        edge_maps = [dict() for _ in range(count)]
        for left in range(count):
            for distance, right in zip(distances[left, 1:], indices[left, 1:]):
                right = int(right)
                distance = float(distance)
                if not distance < translation_limit:
                    continue
                angle = _rotation_angle_deg(rotations[left], rotations[right])
                if not angle < self.max_rotation_deg:
                    continue
                cost = distance / max(translation_scale, 1e-8) + 0.18 * angle
                old = edge_maps[left].get(right)
                if old is None or cost < old[0]:
                    edge_maps[left][right] = (cost, distance, angle)
                    edge_maps[right][left] = (cost, distance, angle)

        retained = [dict() for _ in range(count)]
        for left, edge_map in enumerate(edge_maps):
            ordered = sorted(
                (
                    (cost, right, distance, angle)
                    for right, (cost, distance, angle) in edge_map.items()
                ),
                key=lambda item: item[0],
            )[: self.graph_neighbors]
            for cost, right, distance, angle in ordered:
                retained[left][right] = (cost, distance, angle)
                retained[right][left] = (cost, distance, angle)
        neighbors = [
            sorted(
                (
                    (cost, right, distance, angle)
                    for right, (cost, distance, angle) in edge_map.items()
                ),
                key=lambda item: item[0],
            )
            for edge_map in retained
        ]

        component = np.full(count, -1, dtype=np.int32)
        component_nodes = []
        for start in range(count):
            if component[start] >= 0 or not neighbors[start]:
                continue
            label = len(component_nodes)
            component[start] = label
            stack, nodes = [start], []
            while stack:
                node = stack.pop()
                nodes.append(node)
                for _, nxt, _, _ in neighbors[node]:
                    if component[nxt] < 0:
                        component[nxt] = label
                        stack.append(nxt)
            component_nodes.append(nodes)

        graph = {
            "global_ids": global_ids,
            "global_to_local": {
                int(global_id): local_id
                for local_id, global_id in enumerate(global_ids)
            },
            "rotations": rotations,
            "centers": centers,
            "neighbors": neighbors,
            "component": component,
            "component_nodes": component_nodes,
            "translation_scale": translation_scale,
            "translation_limit": translation_limit,
        }
        self._scene_graph_cache[group] = graph
        return graph

    @staticmethod
    def _turn_penalty(centers: np.ndarray, path, nxt: int) -> float:
        if len(path) < 2:
            return 0.0
        incoming = centers[path[-1]] - centers[path[-2]]
        outgoing = centers[nxt] - centers[path[-1]]
        denominator = np.linalg.norm(incoming) * np.linalg.norm(outgoing)
        if denominator < 1e-8:
            return 0.0
        cosine = np.clip(float(incoming @ outgoing) / denominator, -1.0, 1.0)
        return 1.5 * (1.0 - cosine)

    def _extend_smooth_walk(self, graph, path, target: int, rng):
        neighbors = graph["neighbors"]
        centers = graph["centers"]
        visits = np.zeros(len(neighbors), dtype=np.int32)
        for node in path:
            visits[node] += 1
        while len(path) < int(target):
            current = path[-1]
            if not neighbors[current]:
                return None
            scored = []
            for edge_cost, nxt, _, _ in neighbors[current]:
                repeat_cost = 10.0 * visits[nxt]
                immediate_backtrack = (
                    10.0
                    if len(path) > 1 and nxt == path[-2] and len(neighbors[current]) > 1
                    else 0.0
                )
                scored.append(
                    (
                        edge_cost
                        + repeat_cost
                        + immediate_backtrack
                        + self._turn_penalty(centers, path, nxt),
                        int(nxt),
                    )
                )
            scored.sort(key=lambda item: item[0])
            pool = scored[: min(3, len(scored))]
            costs = np.asarray([item[0] for item in pool], dtype=np.float64)
            probabilities = np.exp(-(costs - costs.min()))
            probabilities /= probabilities.sum()
            nxt = pool[int(rng.choice(len(pool), p=probabilities))][1]
            path.append(nxt)
            visits[nxt] += 1
        return path

    def _generate_sequence(self, group: str, start_global: int, num_views: int, rng):
        graph = self._build_group_graph(group)
        if graph is None or not graph["component_nodes"]:
            return None
        components = graph["component_nodes"]
        component = graph["component"]
        start = graph["global_to_local"].get(int(start_global), -1)
        start_label = int(component[start]) if start >= 0 and component[start] >= 0 else -1
        largest_label = int(np.argmax([len(nodes) for nodes in components]))
        if start_label < 0 or len(components[start_label]) < self.min_component_views:
            chosen_nodes = components[largest_label]
            if len(chosen_nodes) < self.min_component_views:
                return None
            if start >= 0:
                distances = np.linalg.norm(
                    graph["centers"][chosen_nodes] - graph["centers"][start], axis=1
                )
                start = int(chosen_nodes[int(np.argmin(distances))])
            else:
                start = int(chosen_nodes[int(rng.integers(0, len(chosen_nodes)))])

        node_count = len(components[int(component[start])])
        unique_target = min(int(num_views), node_count)
        path = self._unique_beam_path(graph, start, unique_target, rng)
        if len(path) < num_views:
            if not self.allow_repeat:
                return None
            path = self._extend_smooth_walk(graph, path, num_views, rng)
            if path is None:
                return None
        unique_ratio = len(set(path)) / float(num_views)
        if unique_ratio < self.min_unique_ratio:
            return None

        for left, right in zip(path[:-1], path[1:]):
            angle = _rotation_angle_deg(
                graph["rotations"][left], graph["rotations"][right]
            )
            distance = float(
                np.linalg.norm(graph["centers"][left] - graph["centers"][right])
            )
            if not angle < self.max_rotation_deg:
                raise AssertionError("HyperSim pose graph exceeded rotation limit")
            if not distance < graph["translation_limit"]:
                raise AssertionError("HyperSim pose graph exceeded translation limit")
        return path

    def _try_group(self, group: str, first_global: int, num_views: int, rng):
        if group in self._runtime_failed_groups:
            return None
        global_ids = self._group_global_ids(group)
        starts = [int(first_global)]
        remaining = min(self.sequence_start_retries - 1, max(0, len(global_ids) - 1))
        if remaining:
            alternatives = global_ids[global_ids != int(first_global)]
            starts.extend(
                int(value)
                for value in rng.choice(alternatives, remaining, replace=False)
            )
        for start in starts:
            path = self._generate_sequence(group, start, num_views, rng)
            if path is not None:
                return path
        self._runtime_failed_groups.add(group)
        return None

    def _sample_sequence(self, idx: int, num_views: int, rng):
        sample_index = int(idx) // 10
        start_global = int(self.start_img_ids[sample_index])
        initial_scene_id = int(self.sceneids[start_global])
        initial_group = self._scene_id_to_group[initial_scene_id]
        candidates = [(initial_group, start_global)]
        other_groups = [
            group
            for group in self._eligible_groups
            if group != initial_group and group not in self._runtime_failed_groups
        ]
        remaining = min(self.sequence_scene_retries - 1, len(other_groups))
        if remaining:
            for position in np.atleast_1d(
                rng.choice(len(other_groups), remaining, replace=False)
            ):
                group = other_groups[int(position)]
                ids = self._group_global_ids(group)
                candidates.append((group, int(ids[int(rng.integers(0, len(ids)))])))
        failed = []
        for group, start in candidates:
            path = self._try_group(group, start, num_views, rng)
            if path is not None:
                graph = self._build_group_graph(group)
                return sample_index, group, graph["global_ids"][path]
            failed.append(group)
        raise RuntimeError(
            f"HyperSimSeq could not form {num_views} smooth views after "
            f"trying {len(failed)} scenes: {failed[:8]}"
        )

    # Reuse the released diverse-suffix beam search; costs and turn penalties
    # remain HyperSim-specific through the methods and graph defined above.
    _unique_beam_path = UnrealStereo4KSeq._unique_beam_path

    def _get_views(self, idx, resolution, rng, num_views, preserve_fov, sequence_aug):
        if num_views < 1:
            raise ValueError("num_views must be positive")
        _, _, image_indices = self._sample_sequence(idx, num_views, rng)
        views, raw_cache = [], {}
        for view_idx in image_indices:
            view_idx = int(view_idx)
            scene_dir = self.root / self.scenes[self.sceneids[view_idx]]
            rgb_name = self.images[view_idx]
            if view_idx not in raw_cache:
                image = image_open(scene_dir / rgb_name)
                depth = np.asarray(np_load(
                    scene_dir / rgb_name.replace("rgb.png", "depth.npy")
                ), dtype=np.float32).copy()
                if depth.ndim != 2 or depth.shape != (image.height, image.width):
                    raise ValueError(f"RGB/depth shape mismatch: {scene_dir}/{rgb_name}")
                depth[~np.isfinite(depth)] = 0
                with np_load(scene_dir / rgb_name.replace("rgb.png", "cam.npz")) as camera:
                    K = np.asarray(camera["intrinsics"], dtype=np.float32).copy()
                    pose = np.asarray(camera["pose"], dtype=np.float32).copy()
                if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
                    raise ValueError(f"Invalid HyperSim intrinsics: {scene_dir}/{rgb_name}")
                if pose.shape != (4, 4) or not np.isfinite(pose).all():
                    raise ValueError(f"Invalid HyperSim pose: {scene_dir}/{rgb_name}")
                raw_cache[view_idx] = image, depth, K, pose
            raw_image, raw_depth, raw_K, raw_pose = raw_cache[view_idx]
            image, depth, K = self._crop_resize_if_necessary(
                raw_image.copy(), raw_depth.copy(), raw_K.copy(), resolution, rng,
                preserve_fov=preserve_fov, sequence_aug=sequence_aug,
                info=str(scene_dir / rgb_name),
            )
            views.append(dict(
                img=image, depthmap=depth, camera_intrinsics=K,
                camera_pose=raw_pose.copy(), dataset=self.dataset_name,
                label=f"{self.scenes[self.sceneids[view_idx]]}/{rgb_name[:-8]}",
            ))
        if len(views) != num_views:
            raise RuntimeError(f"expected {num_views} views, got {len(views)}")
        return views
