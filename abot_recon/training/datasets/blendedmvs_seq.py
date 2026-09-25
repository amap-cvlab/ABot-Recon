# Adapted from CUT3R and the development loaders; see THIRD_PARTY_NOTICES.md.
# CUT3R-derived portions retain CC BY-NC-SA 4.0.
"""CUT3R BlendedMVS payloads sampled through the development pose graph."""

from __future__ import annotations

import json
import os.path as osp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .base import MultiViewDataset
from .io import image_open, imread_cv2, np_load
from .unreal4k_seq import _rotation_angle_deg


# The development loader identifies this scene as consistently stored sideways.
_KNOWN_SIDEWAYS_SCENES = frozenset({"000000000000000000000012"})


def _decode_name(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _opencv_supports_exr():
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("OpenEXR:"):
            return not line.split(":", 1)[1].strip().upper().startswith("NO")
    return None


class BlendedMVSSeq(MultiViewDataset):
    """Read scene/NNNNNNNN.{jpg,exr,npz} plus official new_overlap.h5.

    Only HDF5 basenames are consumed, not overlap scores. Poses are stored
    as R_cam2world and t_cam2world; depth and translation share the original
    non-metric scale. The producer already inverted the raw world-to-camera
    poses. The split argument does not filter scenes.

    scene_blacklist_path=None or "auto" loads root/blendedmvs_scene_blacklist.json
    when present, otherwise the bundled 38-scene training blacklist. Explicit
    local paths are root-relative unless absolute, accepting JSON lists,
    {"drop_scenes": []}, or .txt files with one scene per line (blank lines and
    # comments ignored). Pass "" to disable this additional scene blacklist.
    Optional upright_mask_path accepts {"drop_scenes": [], "invalid": {scene:
    [basename]}} or {"scenes": {scene: {"drop": bool, "invalid": []}}}.
    Upright masks are loaded only when explicitly supplied. Pose-roll filtering
    and the built-in sideways scene remain active even when the additional
    blacklist is disabled. Graph caches stay in process RAM.
    """

    dataset_name = "blendedmvs_seq"

    def __init__(
        self, root, *, split="train", allow_repeat=True, overlap_path=None,
        scene_blacklist_path=None, upright_mask_path=None,
        max_rotation_deg=40.0, max_translation_factor=5.0,
        pose_knn=24, graph_neighbors=14, beam_width=96,
        min_unique_ratio=0.75, max_abs_roll_deg=45.0, pose_load_workers=1,
        sequence_start_retries=3, sequence_scene_retries=64,
        sideways_scene_ids=(), **kwargs,
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
        self.max_abs_roll_deg = float(max_abs_roll_deg)
        if not np.isfinite(self.max_abs_roll_deg) or not 0 <= self.max_abs_roll_deg <= 180:
            raise ValueError("max_abs_roll_deg must be finite and in [0, 180]")
        self.pose_load_workers = max(1, int(pose_load_workers))
        self.sequence_start_retries = max(1, int(sequence_start_retries))
        self.sequence_scene_retries = max(1, int(sequence_scene_retries))
        self.sideways_scene_ids = set(_KNOWN_SIDEWAYS_SCENES)
        self.sideways_scene_ids.update(str(value) for value in sideways_scene_ids)
        self._invalid_orientation = {}
        self._scene_graph_cache = {}
        self._runtime_failed_scenes = set()
        self._opencv_exr_available = _opencv_supports_exr()
        super().__init__(root=root, split=split, allow_repeat=allow_repeat, **kwargs)
        self.split = split
        if self.num_views < 1:
            raise ValueError("num_views must be positive")
        self.overlap_path = self._local_path(overlap_path or "new_overlap.h5")
        self._load_scene_blacklist(scene_blacklist_path)
        self._load_upright_mask(upright_mask_path)
        self._load_data()

    def _local_path(self, path):
        path = Path(path).expanduser()
        return path if path.is_absolute() else self.root / path

    def _read_json(self, path):
        with self._local_path(path).open(encoding="utf-8") as handle:
            return json.load(handle)

    def _load_scene_blacklist(self, path):
        if path is None or path == "auto":
            root_sidecar = self.root / "blendedmvs_scene_blacklist.json"
            path = root_sidecar.resolve() if root_sidecar.is_file() else (
                Path(__file__).resolve().parent / "metadata" / "blendedmvs_blacklist.txt"
            )
        if not path:
            return
        path = self._local_path(path).resolve()
        if path.suffix.lower() == ".txt":
            with path.open(encoding="utf-8") as handle:
                scenes = [line for raw in handle if (line := raw.strip())
                          and not line.startswith("#")]
        else:
            data = self._read_json(path)
            scenes = data if isinstance(data, list) else data.get("drop_scenes", []) if isinstance(data, dict) else None
        if not isinstance(scenes, list):
            raise ValueError("scene blacklist must be a list or contain drop_scenes")
        self.sideways_scene_ids.update(str(scene) for scene in scenes)

    def _load_upright_mask(self, path):
        if not path:
            return
        data = self._read_json(path)
        if not isinstance(data, dict):
            raise ValueError("upright orientation mask must be a JSON object")
        self.sideways_scene_ids.update(str(value) for value in data.get("drop_scenes", []))
        for scene, names in data.get("invalid", {}).items():
            self._invalid_orientation[str(scene)] = {
                Path(str(name)).stem for name in names
            }
        for scene, record in data.get("scenes", {}).items():
            if record.get("drop", False):
                self.sideways_scene_ids.add(str(scene))
            self._invalid_orientation.setdefault(str(scene), set()).update(
                Path(str(name)).stem for name in record.get("invalid", [])
            )

    def _load_data(self):
        import h5py

        if not self.overlap_path.is_file():
            raise FileNotFoundError(
                f"Missing {self.overlap_path}; download CUT3R's separate new_overlap.h5 artifact"
            )
        self.data_dict, self.all_ref_imgs = {}, []
        with h5py.File(self.overlap_path, "r") as handle:
            for scene in handle:
                if scene in self.sideways_scene_ids:
                    continue
                basenames = handle[scene]["basenames"][:]
                invalid = self._invalid_orientation.get(scene, set())
                indices = [index for index, name in enumerate(basenames)
                           if _decode_name(name) not in invalid]
                if not indices:
                    continue
                self.data_dict[scene] = {"basenames": basenames, "score_matrix": None}
                self.all_ref_imgs.extend((scene, index) for index in indices)
        self.num_imgs = len(self.all_ref_imgs)
        self._eligible_scenes = tuple(sorted(self.data_dict))

    def __len__(self):
        return len(self.all_ref_imgs)

    def get_image_num(self):
        return self.num_imgs

    def _payload_root(self):
        return str(self.root)

    @staticmethod
    def _read_pose(path):
        with np_load(path) as camera:
            rotation = np.asarray(camera["R_cam2world"], dtype=np.float32).copy()
            center = np.asarray(camera["t_cam2world"], dtype=np.float32).copy()
        if (rotation.shape != (3, 3) or center.shape != (3,)
                or not np.isfinite(rotation).all() or not np.isfinite(center).all()):
            raise ValueError(f"Invalid BlendedMVS pose: {path}")
        return rotation, center

    def _read_depth(self, path):
        """Read actual float EXR, retaining the optional development fallback."""
        if self._opencv_exr_available is not False:
            try:
                return imread_cv2(path, cv2.IMREAD_UNCHANGED)
            except (cv2.error, OSError):
                pass
        try:
            import OpenEXR
        except ImportError as error:
            raise OSError(
                f"Cannot decode {path}; use OpenCV with OpenEXR support or install OpenEXR"
            ) from error
        exr = OpenEXR.File(str(path), separate_channels=True)
        channels = exr.channels()
        for preferred in ("Y", "Z", "R"):
            if preferred in channels:
                depth = channels[preferred].pixels
                break
        else:
            if len(channels) != 1:
                raise ValueError(f"cannot choose a depth channel from {sorted(channels)}")
            depth = next(iter(channels.values())).pixels
        return np.asarray(depth, dtype=np.float32).copy()

    def _pose_upright_mask(self, rotations: np.ndarray) -> np.ndarray:
        """Reject roll outliers relative to the dominant per-scene camera up."""
        camera_up = -rotations[:, :, 1]
        _, eigenvectors = np.linalg.eigh(camera_up.T @ camera_up)
        world_up = eigenvectors[:, -1]
        if np.median(camera_up @ world_up) < 0:
            world_up = -world_up

        up_in_camera = np.einsum(
            "nij,j->ni", np.transpose(rotations, (0, 2, 1)), world_up
        )
        projection = np.linalg.norm(up_in_camera[:, :2], axis=1)
        roll = np.degrees(
            np.arctan2(up_in_camera[:, 0], -up_in_camera[:, 1])
        )
        reliable = projection > 0.30
        return ~(reliable & (np.abs(roll) > self.max_abs_roll_deg))

    def _build_scene_graph(self, scene):
        if scene in self._scene_graph_cache:
            return self._scene_graph_cache[scene]
        if scene in self.sideways_scene_ids:
            return None

        basenames = self.data_dict[scene]["basenames"]
        names = [_decode_name(x) for x in basenames]
        scene_dir = osp.join(self._payload_root(), scene)
        paths = [osp.join(scene_dir, name + ".npz") for name in names]

        if self.pose_load_workers > 1:
            with ThreadPoolExecutor(max_workers=self.pose_load_workers) as pool:
                poses = list(pool.map(self._read_pose, paths))
        else:
            poses = [self._read_pose(path) for path in paths]

        rotations = np.stack([x[0] for x in poses])
        centers = np.stack([x[1] for x in poses])
        valid = self._pose_upright_mask(rotations)
        invalid_names = self._invalid_orientation.get(scene, set())
        if invalid_names:
            valid &= np.asarray([name not in invalid_names for name in names])

        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) < 2:
            self._scene_graph_cache[scene] = None
            return None

        valid_centers = centers[valid_indices]
        query_k = min(self.pose_knn + 1, len(valid_indices))
        tree = cKDTree(valid_centers)
        distances, local_neighbors = tree.query(valid_centers, k=query_k)
        if query_k == 1:
            distances = distances[:, None]
            local_neighbors = local_neighbors[:, None]

        nearest = distances[:, 1] if query_k > 1 else np.ones(len(valid_indices))
        positive = nearest[nearest > 1e-8]
        translation_scale = float(np.median(positive)) if len(positive) else 1.0

        edge_maps = [dict() for _ in names]
        for local_i, global_i in enumerate(valid_indices):
            for distance, local_j in zip(
                distances[local_i, 1:], local_neighbors[local_i, 1:]
            ):
                global_j = int(valid_indices[int(local_j)])
                angle = _rotation_angle_deg(rotations[global_i], rotations[global_j])
                # Strict constraint requested by the user.
                if not angle < self.max_rotation_deg:
                    continue
                if not float(distance) < self.max_translation_factor * translation_scale:
                    continue
                cost = float(distance / max(translation_scale, 1e-8)) + 0.08 * angle
                previous = edge_maps[global_i].get(global_j)
                if previous is None or cost < previous[0]:
                    edge_maps[global_i][global_j] = (cost, float(distance), angle)
                previous = edge_maps[global_j].get(int(global_i))
                if previous is None or cost < previous[0]:
                    edge_maps[global_j][int(global_i)] = (cost, float(distance), angle)

        neighbors = []
        for edge_map in edge_maps:
            ordered = sorted(
                ((cost, j, distance, angle) for j, (cost, distance, angle) in edge_map.items()),
                key=lambda x: x[0],
            )
            neighbors.append(ordered[: self.graph_neighbors])

        # Undirected connected components of the retained smooth graph.
        component = np.full(len(names), -1, dtype=np.int32)
        component_nodes = []
        for start in valid_indices:
            if component[start] >= 0 or not neighbors[start]:
                continue
            label = len(component_nodes)
            stack = [int(start)]
            component[start] = label
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
            "names": names,
            "rotations": rotations,
            "centers": centers,
            "valid": valid,
            "neighbors": neighbors,
            "component": component,
            "component_nodes": component_nodes,
            "translation_scale": translation_scale,
        }
        self._scene_graph_cache[scene] = graph
        return graph

    @staticmethod
    def _turn_penalty(centers, path, nxt):
        if len(path) < 2:
            return 0.0
        previous = centers[path[-1]] - centers[path[-2]]
        upcoming = centers[nxt] - centers[path[-1]]
        norm_previous = np.linalg.norm(previous)
        norm_upcoming = np.linalg.norm(upcoming)
        if norm_previous < 1e-8 or norm_upcoming < 1e-8:
            return 0.0
        cosine = np.clip(
            np.dot(previous, upcoming) / (norm_previous * norm_upcoming), -1.0, 1.0
        )
        return 1.5 * (1.0 - float(cosine))

    def _unique_beam_path(self, graph, start, target, rng):
        neighbors = graph["neighbors"]
        centers = graph["centers"]
        states = [(0.0, (int(start),))]
        best_path = states[0][1]

        for _ in range(1, target):
            expanded = []
            for total_cost, path in states:
                current = path[-1]
                visited = set(path)
                for edge_cost, nxt, _, _ in neighbors[current]:
                    if nxt in visited:
                        continue
                    smoothness = self._turn_penalty(centers, path, nxt)
                    jitter = float(rng.uniform(0.0, 1e-4))
                    expanded.append(
                        (total_cost + edge_cost + smoothness + jitter, path + (nxt,))
                    )
            if not expanded:
                break
            expanded.sort(key=lambda x: x[0])
            states = expanded[: self.beam_width]
            best_path = states[0][1]
        return list(best_path)

    def _extend_with_soft_repeats(self, graph, path, target, rng):
        """Fill a short path while strongly preferring never-visited views."""
        neighbors = graph["neighbors"]
        centers = graph["centers"]
        visits = np.zeros(len(neighbors), dtype=np.int32)
        for node in path:
            visits[node] += 1

        while len(path) < target:
            current = path[-1]
            candidates = neighbors[current]
            if not candidates:
                return None

            scored = []
            for edge_cost, nxt, _, _ in candidates:
                repeat_cost = 0.0 if visits[nxt] == 0 else 12.0 * visits[nxt]
                immediate_backtrack = (
                    20.0 if len(path) > 1 and nxt == path[-2] and len(candidates) > 1 else 0.0
                )
                smoothness = self._turn_penalty(centers, path, nxt)
                scored.append(
                    (
                        edge_cost + repeat_cost + immediate_backtrack + smoothness,
                        nxt,
                    )
                )
            scored.sort(key=lambda x: x[0])
            pool = scored[: min(3, len(scored))]
            costs = np.asarray([x[0] for x in pool], dtype=np.float64)
            probabilities = np.exp(-(costs - costs.min()))
            probabilities /= probabilities.sum()
            choice = int(rng.choice(len(pool), p=probabilities))
            nxt = int(pool[choice][1])
            path.append(nxt)
            visits[nxt] += 1
        return path

    def generate_sequence(
        self, scene, adj_list, num_views, start_index, rng, allow_repeat=False
    ):
        del adj_list
        graph = self._build_scene_graph(scene)
        if graph is None or not graph["component_nodes"]:
            return None

        component = graph["component"]
        components = graph["component_nodes"]
        start_index = int(start_index)
        start_component = int(component[start_index]) if component[start_index] >= 0 else -1

        if start_component < 0:
            chosen_component = int(np.argmax([len(x) for x in components]))
            nodes = components[chosen_component]
            centers = graph["centers"]
            distances = np.linalg.norm(centers[nodes] - centers[start_index], axis=1)
            start_index = int(nodes[int(np.argmin(distances))])
        else:
            chosen_component = start_component
            nodes = components[chosen_component]

        # If the reference lies in a tiny island, use the largest smooth component.
        largest_component = int(np.argmax([len(x) for x in components]))
        if len(nodes) < min(num_views, len(components[largest_component])):
            chosen_component = largest_component
            nodes = components[chosen_component]
            centers = graph["centers"]
            distances = np.linalg.norm(centers[nodes] - centers[start_index], axis=1)
            start_index = int(nodes[int(np.argmin(distances))])

        unique_target = min(int(num_views), len(nodes))
        path = self._unique_beam_path(graph, start_index, unique_target, rng)

        if len(path) < num_views:
            if not allow_repeat:
                return None
            path = self._extend_with_soft_repeats(graph, path, int(num_views), rng)
            if path is None:
                return None

        unique_ratio = len(set(path)) / float(num_views)
        if unique_ratio < min(self.min_unique_ratio, len(nodes) / float(num_views)):
            return None

        # Defensive hard check: no returned edge may reach the configured limit.
        rotations = graph["rotations"]
        for left, right in zip(path[:-1], path[1:]):
            if not _rotation_angle_deg(rotations[left], rotations[right]) < self.max_rotation_deg:
                raise AssertionError("pose graph emitted an invalid rotation transition")
        return path

    def _try_scene_sequence(self, scene, first_start, num_views, rng):
        """Try a few starts before declaring that a scene cannot be sampled."""
        if scene in self._runtime_failed_scenes:
            return None
        basenames = self.data_dict[scene]["basenames"]
        starts = [int(first_start)]
        remaining = min(self.sequence_start_retries - 1, max(0, len(basenames) - 1))
        if remaining:
            candidates = np.asarray(
                [index for index in range(len(basenames)) if index != int(first_start)],
                dtype=np.int64,
            )
            starts.extend(
                int(x) for x in rng.choice(candidates, size=remaining, replace=False)
            )
        for start in starts:
            sequence = self.generate_sequence(
                scene, None, num_views, start, rng, self.allow_repeat
            )
            if sequence is not None:
                return sequence
        self._runtime_failed_scenes.add(scene)
        return None

    def _sample_sequence_with_scene_retry(self, idx, num_views, rng):
        """Return a valid sequence, changing scenes after each scene failure."""
        initial_scene, initial_start = self.all_ref_imgs[int(idx)]
        candidates = [(initial_scene, int(initial_start))]
        other_scenes = [
            scene
            for scene in self._eligible_scenes
            if scene != initial_scene and scene not in self._runtime_failed_scenes
        ]
        remaining = min(self.sequence_scene_retries - 1, len(other_scenes))
        if remaining:
            selected = rng.choice(len(other_scenes), size=remaining, replace=False)
            for position in np.atleast_1d(selected):
                scene = other_scenes[int(position)]
                start = int(rng.integers(0, len(self.data_dict[scene]["basenames"])))
                candidates.append((scene, start))

        failures = []
        for scene, start in candidates:
            if scene in self._runtime_failed_scenes:
                continue
            sequence = self._try_scene_sequence(scene, start, num_views, rng)
            if sequence is not None:
                return scene, sequence
            failures.append(scene)
        raise RuntimeError(
            f"BlendedMVSSeq could not generate {num_views} views after trying "
            f"{len(failures)} scenes; failed scenes: {failures[:8]}"
        )

    def _get_views(self, idx, resolution, rng, num_views, preserve_fov, sequence_aug):
        if num_views < 1:
            raise ValueError("num_views must be positive")
        scene, indices = self._sample_sequence_with_scene_retry(idx, num_views, rng)
        basenames = self.data_dict[scene]["basenames"]
        directory = self.root / scene
        views, raw_cache = [], {}
        for index in indices:
            index = int(index)
            basename = _decode_name(basenames[index])
            if index not in raw_cache:
                image = image_open(directory / (basename + ".jpg"))
                depth = np.asarray(self._read_depth(directory / (basename + ".exr")), dtype=np.float32)
                if depth.ndim != 2 or depth.shape != (image.height, image.width):
                    raise ValueError(f"RGB/depth shape mismatch: {directory}/{basename}")
                with np_load(directory / (basename + ".npz")) as camera:
                    K = np.asarray(camera["intrinsics"], dtype=np.float32).copy()
                    pose = np.eye(4, dtype=np.float32)
                    pose[:3, :3] = camera["R_cam2world"]
                    pose[:3, 3] = camera["t_cam2world"]
                if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
                    raise ValueError(f"Invalid BlendedMVS intrinsics: {directory}/{basename}")
                raw_cache[index] = image, depth, K, pose
            raw_image, raw_depth, raw_K, raw_pose = raw_cache[index]
            depth = raw_depth.copy()
            positive = depth > 0
            if positive.any():
                threshold = np.percentile(depth[positive], 97)
                depth[depth > threshold] = 0
            # Match development percentile selection before the shared finite cleanup.
            depth[~np.isfinite(depth)] = 0
            image, depth, K = self._crop_resize_if_necessary(
                raw_image.copy(), depth, raw_K.copy(), resolution, rng,
                preserve_fov=preserve_fov, sequence_aug=sequence_aug,
                info=str(directory / (basename + ".jpg")),
            )
            views.append(dict(
                img=image, depthmap=depth, camera_intrinsics=K,
                camera_pose=raw_pose.copy(), dataset=self.dataset_name,
                label=f"{scene}/{basename}",
            ))
        if len(views) != num_views:
            raise RuntimeError(f"expected {num_views} views, got {len(views)}")
        return views
