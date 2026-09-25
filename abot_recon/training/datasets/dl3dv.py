import os
import os.path as osp

import cv2
import numpy as np

from .base import MultiViewDataset
from .io import image_open, imread_cv2, np_load


_SEQUENCE_BLACKLIST_FILENAME = "dl3dv_geometry_blacklist_20260810.txt"
_DEFAULT_SEQUENCE_BLACKLIST_PATH = osp.join(
    osp.dirname(__file__),
    "metadata",
    _SEQUENCE_BLACKLIST_FILENAME,
)


_DEPTH_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_DEPTH_ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def _remove_small_depth_regions(depthmap):
    """Conservatively remove thin/small valid regions without filling depth holes."""
    valid = depthmap > 0
    valid = cv2.morphologyEx(
        valid.astype(np.uint8), cv2.MORPH_OPEN, _DEPTH_OPEN_KERNEL
    )
    valid = cv2.erode(valid, _DEPTH_ERODE_KERNEL)
    depthmap[(valid == 0) | (depthmap <= 0)] = 0
    return depthmap


class DL3DV(MultiViewDataset):
    dataset_name = "dl3dv"
    def __init__(
        self,
        root,
        *args,
        sequence_blacklist_path="auto",
        min_interval=1,
        max_interval=20,
        **kwargs,
    ):
        self.video = True
        self.min_interval = int(min_interval)
        self.max_interval = int(max_interval)
        self.is_metric = False
        self.sequence_blacklist_path = self._resolve_sequence_blacklist_path(
            root, sequence_blacklist_path
        )
        self.excluded_sequences = self._load_sequence_blacklist(
            self.sequence_blacklist_path
        )
        super().__init__(root=root, *args, **kwargs)

        self.loaded_data = self._load_data()

    @classmethod
    def _resolve_sequence_blacklist_path(cls, root, path):
        """Resolve ``auto`` to a dataset-root or packaged blacklist."""
        if path != "auto":
            return path

        root_path = str(root).rstrip("/")
        dataset_path = osp.join(root_path, _SEQUENCE_BLACKLIST_FILENAME)
        if osp.isfile(dataset_path):
            return dataset_path

        if osp.isfile(_DEFAULT_SEQUENCE_BLACKLIST_PATH):
            print(
                "DL3DV: dataset-root blacklist unavailable; "
                f"using packaged fallback {_DEFAULT_SEQUENCE_BLACKLIST_PATH}"
            )
            return _DEFAULT_SEQUENCE_BLACKLIST_PATH
        raise FileNotFoundError(
            "DL3DV default sequence blacklist is missing from both the dataset "
            "root and the installed package; reinstall the package or set "
            'sequence_blacklist_path="" to explicitly disable filtering.'
        )

    @staticmethod
    def _load_sequence_blacklist(path):
        """Load exact ``<bucket>/<scene>`` keys, one per text line."""
        if not path:
            return set()
        path = osp.expanduser(str(path))
        if not osp.isfile(path):
            raise FileNotFoundError(f"DL3DV sequence blacklist does not exist: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        excluded = {
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        print(
            f"DL3DV: loaded {len(excluded)} excluded sequences from {path}"
        )
        return excluded

    def _load_data(self):
        print(f"Loading DL3DV dataset from {self.ROOT}")
        self.all_scenes = sorted(
            [f for f in os.listdir(self.ROOT) if os.path.isdir(osp.join(self.ROOT, f))]
        )
        subscenes = []
        for scene in self.all_scenes:
            # not empty
            subscenes.extend(
                [
                    osp.join(scene, f)
                    for f in os.listdir(osp.join(self.ROOT, scene))
                    if os.path.isdir(osp.join(self.ROOT, scene, f))
                    and len(os.listdir(osp.join(self.ROOT, scene, f))) > 0
                ]
            )

        excluded_present = self.excluded_sequences.intersection(subscenes)
        subscenes = [
            scene for scene in subscenes
            if scene not in self.excluded_sequences
        ]
        missing_exclusions = self.excluded_sequences.difference(excluded_present)
        print(
            "DL3DV: excluded "
            f"{len(excluded_present)}/{len(self.excluded_sequences)} blacklisted "
            f"sequences; {len(missing_exclusions)} blacklist entries are absent "
            "from this root"
        )

        offset = 0
        scenes = []
        sceneids = []
        images = []
        scene_img_list = []
        start_img_ids = []
        j = 0
        print(f"Loading DL3DV dataset with {len(subscenes)} scenes from {self.ROOT}")
        for scene_idx, scene in enumerate(subscenes):
            scene_dir = osp.join(self.ROOT, scene, "dense")
            rgb_paths = sorted(
                [
                    f
                    for f in os.listdir(os.path.join(scene_dir, "rgb"))
                    if f.endswith(".png")
                ]
            )
            assert len(rgb_paths) > 0, f"{scene_dir} is empty."
            num_imgs = len(rgb_paths)
            cut_off = (
                self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            )

            if num_imgs < cut_off:
                print(f"Skipping dl3dv {scene}")
                continue

            img_ids = list(np.arange(num_imgs) + offset)
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

            scenes.append(scene)
            scene_img_list.append(img_ids)
            sceneids.extend([j] * num_imgs)
            images.extend(rgb_paths)
            start_img_ids.extend(start_img_ids_)
            offset += num_imgs
            j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list
        
        print("Loaded DL3DV dataset done.")

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(
        self, idx, resolution, rng, num_views, preserve_fov, sequence_aug
    ):
        start_id = self.start_img_ids[idx]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]
        remaining = len(all_image_ids) - 1 - all_image_ids.index(start_id)
        adaptive_min, adaptive_max = self.adaptive_interval_bounds(
            remaining,
            num_views,
            getattr(self, "min_interval", 1),
            self.max_interval,
        )
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,
            min_interval=adaptive_min,
            max_interval=adaptive_max,
            block_shuffle=25,
            video_prob=1., # just for long seq train
        )
        image_idxs = np.array(all_image_ids)[pos]

        views = []
        for view_idx in image_idxs:
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id], "dense")

            rgb_path = self.images[view_idx]
            basename = rgb_path[:-4]

            rgb_image = image_open(
                osp.join(scene_dir, "rgb", rgb_path)
            )
            depthmap = np_load(osp.join(scene_dir, "depth", basename + ".npy")).astype(
                np.float32
            )
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            cam_file = np_load(osp.join(scene_dir, "cam", basename + ".npz"))
            sky_mask = (
                imread_cv2(
                    osp.join(scene_dir, "sky_mask", rgb_path), cv2.IMREAD_UNCHANGED
                )
                >= 127
            )
            outlier_mask = imread_cv2(
                osp.join(scene_dir, "outlier_mask", rgb_path), cv2.IMREAD_UNCHANGED
            )
            depthmap[sky_mask] = -1.0
            depthmap[outlier_mask >= 127] = 0.0
            depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)
            threshold = (
                np.percentile(depthmap[depthmap > 0], 98)
                if depthmap[depthmap > 0].size > 0
                else 0
            )
            depthmap[depthmap > threshold] = 0.0
            depthmap = _remove_small_depth_regions(depthmap)

            intrinsics = cam_file["intrinsic"].astype(np.float32)
            camera_pose = cam_file["pose"].astype(np.float32)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image,
                depthmap,
                intrinsics,
                resolution,
                rng=rng,
                preserve_fov=preserve_fov,
                sequence_aug=sequence_aug,
                info=view_idx,
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="dl3dv",
                    label=self.scenes[scene_id] + "_" + rgb_path,
                    instance=osp.join(scene_dir, "rgb", rgb_path),
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(0.9, dtype=np.float32),
                    img_mask=True,
                    ray_mask=False,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                )
            )
        assert len(views) == num_views
        return views
