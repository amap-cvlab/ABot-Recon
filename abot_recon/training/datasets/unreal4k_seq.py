# Adapted from CUT3R and the development loaders; see THIRD_PARTY_NOTICES.md.
# CUT3R-derived portions retain CC BY-NC-SA 4.0.
"""UnrealStereo4K CUT3R payloads sampled through a pose-neighbour graph.

Numeric frame order is not a trajectory. The development implementation's
KNN edges, strict motion limits, diverse beam search, and soft revisits are
preserved; graphs are lazy, per-process RAM caches, never disk sidecars.
"""

from __future__ import annotations

import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

from .base import MultiViewDataset
from .io import imread_cv2, np_load


# Inherited CUT3R world-axis conversion; input cam2world is already C2W.
R_CONV = np.array(
    [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)


def _rotation_angle_deg(left, right):
    cosine = np.clip((np.trace(left.T @ right) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


class UnrealStereo4KSeq(MultiViewDataset):
    dataset_name = "unreal4k_seq"

    def __init__(
        self, root, *, split=None, max_rotation_deg=20.0,
        max_translation_factor=5.0, pose_knn=64, graph_neighbors=24,
        beam_width=128, min_unique_ratio=0.75, pose_load_workers=1,
        sequence_start_retries=6, sequence_scene_retries=18, **kwargs,
    ):
        if split is not None:
            raise ValueError("UnrealStereo4KSeq uses split=None")
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
        self.pose_load_workers = max(1, int(pose_load_workers))
        self.sequence_start_retries = max(1, int(sequence_start_retries))
        self.sequence_scene_retries = max(1, int(sequence_scene_retries))
        self._scene_graph_cache = {}
        self._runtime_failed_scenes = set()
        super().__init__(root=root, split=split, **kwargs)
        self.split = None
        self._load_data()

    def _load_data(self):
        self.scenes, self.sceneids, self.images = [], [], []
        self.start_img_ids, self.scene_img_list = [], []
        cutoff = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
        for scene in sorted(path for path in self.root.iterdir() if path.is_dir()):
            for mode in ("0", "1"):
                directory = scene / mode
                names = sorted(name[:-8] for name in os.listdir(directory) if name.endswith("_rgb.png"))
                if len(names) < cutoff:
                    continue
                ids = list(range(len(self.images), len(self.images) + len(names)))
                self.sceneids.extend([len(self.scenes)] * len(names))
                self.scenes.append(str(directory))
                self.images.extend(names)
                self.scene_img_list.append(ids)
                self.start_img_ids.extend(ids[: len(names) - cutoff + 1])

    def __len__(self):
        # Preserve the source adapter's inventory multiplicity.
        return 10 * len(self.start_img_ids)

    @staticmethod
    def _read_pose(path: str) -> np.ndarray:
        params = np_load(path)
        try:
            return np.asarray(params["cam2world"], dtype=np.float32).copy()
        finally:
            close = getattr(params, "close", None)
            if close is not None:
                close()

    def _build_scene_graph(self, scene_id: int):
        scene_id = int(scene_id)
        if scene_id in self._scene_graph_cache:
            return self._scene_graph_cache[scene_id]

        global_ids = np.asarray(self.scene_img_list[scene_id], dtype=np.int64)
        names = [self.images[int(index)] for index in global_ids]
        scene_dir = self.scenes[scene_id]
        paths = [osp.join(scene_dir, name + ".npz") for name in names]
        if self.pose_load_workers > 1:
            with ThreadPoolExecutor(max_workers=self.pose_load_workers) as pool:
                poses = np.stack(list(pool.map(self._read_pose, paths)))
        else:
            poses = np.stack([self._read_pose(path) for path in paths])

        rotations = poses[:, :3, :3]
        centers = poses[:, :3, 3]
        count = len(centers)
        if count < 2:
            self._scene_graph_cache[scene_id] = None
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
                cost = distance / max(translation_scale, 1e-8) + 0.16 * angle
                old = edge_maps[left].get(right)
                if old is None or cost < old[0]:
                    edge_maps[left][right] = (cost, distance, angle)
                    edge_maps[right][left] = (cost, distance, angle)

        # Keep each camera's best transitions, then restore symmetry.  This
        # bounds path-search work without turning an undirected edge one-way.
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
            stack = [start]
            nodes = []
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
            "names": names,
            "rotations": rotations,
            "centers": centers,
            "neighbors": neighbors,
            "component": component,
            "component_nodes": component_nodes,
            "translation_scale": translation_scale,
            "translation_limit": translation_limit,
        }
        self._scene_graph_cache[scene_id] = graph
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
        return 2.0 * (1.0 - cosine)

    def _unique_beam_path(self, graph, start: int, target: int, rng):
        neighbors = graph["neighbors"]
        centers = graph["centers"]
        states = [(0.0, (int(start),))]
        best = states[0][1]
        for _ in range(1, int(target)):
            expanded = []
            for total_cost, path in states:
                visited = set(path)
                current = path[-1]
                for edge_cost, nxt, _, _ in neighbors[current]:
                    if nxt in visited:
                        continue
                    score = (
                        total_cost
                        + edge_cost
                        + self._turn_penalty(centers, path, nxt)
                        + float(rng.uniform(0.0, 1e-4))
                    )
                    expanded.append((score, path + (int(nxt),)))
            if not expanded:
                break
            expanded.sort(key=lambda item: item[0])
            # Preserve paths that arrive through different final edges.  This
            # prevents all beam slots collapsing onto one local corridor.
            diverse = []
            seen_suffixes = set()
            for state in expanded:
                path = state[1]
                suffix = path[-2:]
                if suffix in seen_suffixes:
                    continue
                seen_suffixes.add(suffix)
                diverse.append(state)
                if len(diverse) >= self.beam_width:
                    break
            states = diverse
            best = states[0][1]
        return list(best)

    def _extend_with_soft_repeats(self, graph, path, target: int, rng):
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
                repeat = 12.0 * visits[nxt]
                backtrack = (
                    12.0
                    if len(path) > 1 and nxt == path[-2] and len(neighbors[current]) > 1
                    else 0.0
                )
                scored.append(
                    (
                        edge_cost
                        + repeat
                        + backtrack
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

    def _generate_sequence(self, scene_id: int, start: int, num_views: int, rng):
        graph = self._build_scene_graph(scene_id)
        if graph is None or not graph["component_nodes"]:
            return None
        components = graph["component_nodes"]
        component = graph["component"]
        largest_label = int(np.argmax([len(nodes) for nodes in components]))
        start = int(start)
        start_label = int(component[start]) if component[start] >= 0 else -1
        if start_label < 0 or len(components[start_label]) < num_views:
            chosen_nodes = components[largest_label]
            if len(chosen_nodes) < num_views and not self.allow_repeat:
                return None
            distances = np.linalg.norm(
                graph["centers"][chosen_nodes] - graph["centers"][start], axis=1
            )
            start = int(chosen_nodes[int(np.argmin(distances))])

        target = min(int(num_views), len(components[int(component[start])]))
        path = self._unique_beam_path(graph, start, target, rng)
        if len(path) < num_views:
            if not self.allow_repeat:
                return None
            path = self._extend_with_soft_repeats(graph, path, num_views, rng)
            if path is None:
                return None
        if len(set(path)) / float(num_views) < min(
            self.min_unique_ratio,
            len(components[int(component[start])]) / float(num_views),
        ):
            return None

        for left, right in zip(path[:-1], path[1:]):
            angle = _rotation_angle_deg(
                graph["rotations"][left], graph["rotations"][right]
            )
            distance = float(
                np.linalg.norm(graph["centers"][left] - graph["centers"][right])
            )
            if not angle < self.max_rotation_deg:
                raise AssertionError("Unreal4K pose graph exceeded rotation limit")
            if not distance < graph["translation_limit"]:
                raise AssertionError("Unreal4K pose graph exceeded translation limit")
        return path

    def _try_scene_sequence(self, scene_id: int, first_start: int, num_views: int, rng):
        if scene_id in self._runtime_failed_scenes:
            return None
        count = len(self.scene_img_list[scene_id])
        starts = [int(first_start)]
        remaining = min(self.sequence_start_retries - 1, max(0, count - 1))
        if remaining:
            candidates = np.asarray(
                [index for index in range(count) if index != int(first_start)],
                dtype=np.int64,
            )
            starts.extend(int(x) for x in rng.choice(candidates, remaining, replace=False))
        for start in starts:
            path = self._generate_sequence(scene_id, start, num_views, rng)
            if path is not None:
                return path
        self._runtime_failed_scenes.add(scene_id)
        return None

    def _sample_sequence(self, idx: int, num_views: int, rng):
        start_global = int(self.start_img_ids[int(idx) // 10])
        initial_scene = int(self.sceneids[start_global])
        initial_ids = self.scene_img_list[initial_scene]
        initial_start = int(initial_ids.index(start_global))
        candidates = [(initial_scene, initial_start)]
        other_scenes = [
            scene_id
            for scene_id in range(len(self.scenes))
            if scene_id != initial_scene and scene_id not in self._runtime_failed_scenes
        ]
        remaining = min(self.sequence_scene_retries - 1, len(other_scenes))
        if remaining:
            selected = rng.choice(len(other_scenes), remaining, replace=False)
            for position in np.atleast_1d(selected):
                scene_id = other_scenes[int(position)]
                start = int(rng.integers(0, len(self.scene_img_list[scene_id])))
                candidates.append((scene_id, start))
        failed = []
        for scene_id, start in candidates:
            path = self._try_scene_sequence(scene_id, start, num_views, rng)
            if path is not None:
                return scene_id, path
            failed.append(self.scenes[scene_id])
        raise RuntimeError(
            f"UnReal4KSeq could not form {num_views} smooth views after "
            f"trying {len(failed)} scene/mode groups: {failed[:6]}"
        )

    def _get_views(self, idx, resolution, rng, num_views, preserve_fov, sequence_aug):
        scene_id, local_indices = self._sample_sequence(idx, num_views, rng)
        graph = self._build_scene_graph(scene_id)
        scene_dir = self.scenes[scene_id]
        views, raw_cache = [], {}
        for local_index in local_indices:
            local_index = int(local_index)
            basename = graph["names"][local_index]
            if local_index not in raw_cache:
                image = imread_cv2(osp.join(scene_dir, basename + "_rgb.png"))
                depth = np.asarray(np_load(osp.join(scene_dir, basename + "_depth.npy")), dtype=np.float32).copy()
                with np_load(osp.join(scene_dir, basename + ".npz")) as camera:
                    K = np.asarray(camera["intrinsics"], dtype=np.float32).copy()
                    pose = R_CONV @ np.asarray(camera["cam2world"], dtype=np.float32)
                raw_cache[local_index] = image, depth, K, pose
            raw_image, raw_depth, raw_K, raw_pose = raw_cache[local_index]
            depth = raw_depth.copy()
            depth[depth >= 1000] = -1.0
            positive = depth > 0
            threshold = np.percentile(depth[positive], 98) if positive.any() else 0.0
            depth[depth > threshold] = 0
            image, depth, K = self._crop_resize_if_necessary(
                raw_image.copy(), depth, raw_K.copy(), resolution, rng,
                preserve_fov=preserve_fov, sequence_aug=sequence_aug,
                info=osp.join(scene_dir, basename + "_rgb.png"),
            )
            views.append(dict(
                img=image, depthmap=depth, camera_intrinsics=K,
                camera_pose=raw_pose.copy(), dataset=self.dataset_name,
                label=osp.join(osp.relpath(scene_dir, self.ROOT), basename),
            ))
        if len(views) != num_views:
            raise RuntimeError(f"expected {num_views} views, got {len(views)}")
        return views
