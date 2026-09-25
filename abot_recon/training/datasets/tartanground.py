"""TartanGround loader for the processed long-sequence layout.

==========================================================================
Directory layout
==========================================================================
    ROOT (= scenes_tartanground) /
        <Scene>__omni__P<NNNN>__rcam_<dir> /
            images  / 00000.jpg ... NNNNN.jpg
            depths  / 00000.npy ... (HxW float32, sky pixels with very large depth)
            cameras / 00000.npz   (camera_intrinsics 3x3, camera_pose 4x4 c2w)
            masks   / 00000.npy

==========================================================================
Dataset stats (measured on 441 sequences at the canonical processed root)
==========================================================================
    sequences per scene-seq : min=318  median=1279  max=7407
    >=300 frames : 100%   >=500 : 98%   >=1000 : 66%
    is_metric : True

==========================================================================
Interval recommendation (per-environment, derived from co-visibility)
==========================================================================
TartanGround spans 62 environments with very different motion dynamics.
At a uniform max_interval=8, ~32 envs (53% of scenes) end up with
covis_p25 < 0.30 -- too aggressive for those scenes. We solve this by
detecting the env from the scene name (first '__'-separated token) and
applying per-env bounds. The remaining envs use default (1, 8).

Healthy threshold: covis_p25 >= 0.30 at max_interval. Per-env list of
env -> (min, max) overrides comes from a 60-scene-per-env motion audit;
TG is ground-robot navigation, so per-frame motion depends heavily on
the foreground content. Forest / marsh envs have high foreground
parallax -> covis decays fast. Indoor / dense-urban envs have farther
content on average -> covis decays slowly. The class of environment
explains 60-70% of the per-stride covis variance.

Sequence length is NOT a constraint (median 1279 frames).

==========================================================================
Eval-only mode (held-out scenes for validation)
==========================================================================
Pass ``eval_only=True`` to flip the loader into an include-only mode that
loads ONLY the scenes named in ``include_scenes`` (defaults to
``TG_DEFAULT_EVAL_SCENES`` -- the same 7-10 scenes the training path holds
out). ``exclude_scenes`` is silently ignored in this mode because include
and exclude semantics are mutually exclusive.

Pair with ``max_starts_per_scene=K`` (positive int) to also make the eval
sampling deterministic: each scene yields exactly K starts spaced by
``np.linspace(0, num_imgs - 1, K)`` so ``len(dataset) == K * n_scenes``.
K=1 means "always start at frame 0". With ``max_starts_per_scene=None``
the loader keeps the training-style "every frame is a valid start", which
is rarely what you want for eval.
"""
import os
import os.path as osp

import numpy as np

from .base import MultiViewDataset
from .foldback import foldback_from_start_long
from .io import imread_cv2, np_load


_SUBDIR_IMAGES = "images"
_SUBDIR_DEPTHS = "depths"
_SUBDIR_CAMERAS = "cameras"
_IMG_EXT = ".jpg"
_DEPTH_EXT = ".npy"
_CAM_EXT = ".npz"

_MIN_SCENE_LENGTH = 32
_SKY_DEPTH_THRESHOLD = 1000.0


# Default eight-scene evaluation set:
# - All scenes >= 2560 frames (= 320 frames * stride=8 = 2560-frame span)
# - Covers all 4 motion classes (extremely_fast / fast / moderate / slow)
# - Each scene picked for being the longest in its env (max diversity)
# These are EXCLUDED from training by default (see __init__ exclude_scenes).
TG_DEFAULT_EVAL_SCENES = frozenset({
    # extremely fast (3 scenes) -- forest / marsh / dense parallax
    "ForestEnv__omni__P0010__rcam_front",                 # 3940 frames
    "GreatMarsh__omni__P0004__rcam_front",                # 4288
    # fast (2 scenes) -- urban architecture / alien terrain
    "CastleFortress__omni__P0006__rcam_front",            # 4135
    # moderate (3 scenes) -- mixed indoor/outdoor
    "AbandonedCable__omni__P0003__rcam_front",            # 3234 (outdoor with structures)
    "Hospital__omni__P0005__rcam_front",                  # 3444 (indoor)
    "NordicHarbor__omni__P0005__rcam_front",              # 3641 (coastal)
    # slow / default (2 scenes) -- rich-detail outdoor
    "OldScandinavia__omni__P0006__rcam_front",            # 4397 (old village)
    "FactoryWeather__omni__P0004__rcam_front",            # 4213 (industrial)
})


class TartanGround(MultiViewDataset):
    """TartanGround (reprocessed layout) with foldback sampling."""
    dataset_name = "tartanground"

    # Per-environment (min, max) interval overrides. Env name = first
    # '__'-separated token of the scene directory (e.g.
    # "ForestEnv__omni__P0001__rcam_front" -> "ForestEnv"). Envs not in
    # this dict use the default (min_interval, max_interval).
    
    DEFAULT_INTERVAL_BY_ENV = {
        # extremely fast (forest / marsh -- dense foreground parallax)
        "ForestEnv":                  (1, 4),   # 30 scenes
        "GreatMarsh":                 (1, 4),   # 8
        "SeasonalForestWinterNight":  (1, 4),   # 4
        "SeasonalForestAutumn":       (1, 4),   # 4
        "SeasonalForestSpring":       (1, 4),   # 4
        "SeasonalForestWinter":       (1, 4),   # 4

        # fast
        "CastleFortress":             (1, 6),   # 12
        "OldIndustrialCity":          (1, 6),   # 11
        "BrushifyMoon":               (1, 6),   # 8
        "Gascola":                    (1, 6),   # 8
        "Slaughter":                  (1, 6),   # 8
        "GothicIsland":               (1, 6),   # 5
        "OldTownSummer":              (1, 6),   # 4
        "HQWesternSaloon":            (1, 6),   # 3

        # moderately fast
        "ConstructionSite":           (1, 8),   # 16
        "AbandonedCable":             (1, 8),   # 14
        "Antiquity3D":                (1, 8),   # 12
        "Downtown":                   (1, 8),   # 8
        "MiddleEast":                 (1, 8),   # 8
        "ModularNeighborhoodIntExt":  (1, 8),   # 7
        "Hospital":                   (1, 8),   # 6
        "IndustrialHangar":           (1, 8),   # 6
        "ModularNeighborhood":        (1, 8),   # 6
        "NordicHarbor":               (1, 8),   # 6
        "ModUrbanCity":               (1, 8),   # 5
        "OldBrickHouseNight":         (1, 8),   # 5
        "Rome":                       (1, 8),   # 5
        "SoulCity":                   (1, 8),   # 5
        "Fantasy":                    (1, 8),   # 4
        "SeasonalForestSummerNight":  (1, 8),   # 4
        "HongKong":                   (1, 8),   # 3
        "House":                      (1, 8),   # 3
        # All other envs (~26 envs / 174 scenes) use the default (1, 8).
    }

    def __init__(self, root, *args,
                 min_interval=1, max_interval=8,
                 interval_by_env=None,
                 exclude_scenes=None,
                 eval_only=False,
                 include_scenes=None,
                 max_starts_per_scene=None,
                 forward_only=False, recent_stride_memory=8,
                 fix_interval_prob=0.5,
                 **kwargs):
        self.video = True
        self.is_metric = True
        # Default bounds (used when env is not in interval_by_env)
        self.min_interval = int(min_interval)
        self.max_interval = int(max_interval)
        # Per-env override dict (None -> use the curated default above).
        # Pass {} to disable per-env overrides entirely.
        self.interval_by_env = (
            dict(interval_by_env) if interval_by_env is not None
            else dict(self.DEFAULT_INTERVAL_BY_ENV)
        )
        # ``eval_only`` (default False): when True, this loader becomes a held-out
        # eval set. ``include_scenes`` (None -> TG_DEFAULT_EVAL_SCENES) defines
        # the allowed set; everything else on disk is ignored. ``exclude_scenes``
        # is silently ignored in this mode because include/exclude semantics are
        # mutually exclusive (an include list is already the complete spec).
        self.eval_only = bool(eval_only)
        if self.eval_only:
            self.include_scenes = (
                set(include_scenes) if include_scenes is not None
                else set(TG_DEFAULT_EVAL_SCENES)
            )
            self.exclude_scenes = set()  # not used in eval_only path
        else:
            self.include_scenes = None
            # Scenes to skip during loading. None -> use TG_DEFAULT_EVAL_SCENES
            # (the curated 10-scene val set held out for evaluation). Pass
            # set() / {} / [] to include ALL scenes (e.g. for evaluation
            # itself, or for ablations comparing with-vs-without held-out).
            self.exclude_scenes = (
                set(exclude_scenes) if exclude_scenes is not None
                else set(TG_DEFAULT_EVAL_SCENES)
            )
        # When set to a positive int K, each scene contributes EXACTLY K
        # starting frames spaced via ``np.linspace(0, num_imgs - 1, K)`` so
        # eval iteration order is deterministic and cheap (K=1 -> frame 0
        # only). None preserves the legacy behavior where every frame is a
        # valid start (huge ``len(self)`` but matches training-time sampling).
        self.max_starts_per_scene = (
            int(max_starts_per_scene) if max_starts_per_scene is not None else None
        )
        if self.max_starts_per_scene is not None and self.max_starts_per_scene < 1:
            raise ValueError(
                f"max_starts_per_scene must be a positive int or None, got {max_starts_per_scene!r}"
            )
        # TartanGround is synthetic robot navigation; the world is static
        # so reverse-time playback is geometrically valid. Default
        # forward_only=False matches the source foldback design.
        self.forward_only = bool(forward_only)
        self.recent_stride_memory = int(recent_stride_memory)
        # Used only when forward_only=True. CUT3R-style: probability of
        # using one fixed stride for the whole walk vs per-pair random.
        self.fix_interval_prob = float(fix_interval_prob)
        super().__init__(root=root, *args, **kwargs)
        self._load_data()

    @staticmethod
    def _scene_env(scene_name):
        """Return the env name (first '__'-separated token)."""
        return scene_name.split("__", 1)[0]

    def _load_data(self):
        scene_dirs = sorted(
            d for d in os.listdir(self.ROOT)
            if os.path.isdir(os.path.join(self.ROOT, d))
        )
        print(f"Loading TartanGroundLong dataset with {len(scene_dirs)} scenes from {self.ROOT}")

        offset = 0
        scenes = []
        sceneids = []
        images = []
        scene_img_list = []
        start_img_ids = []
        scene_envs = []          # parallel to `scenes`: env name string
        scene_index = 0
        env_counter = {}

        n_excluded = 0
        n_not_in_include = 0
        for scene in scene_dirs:
            if self.eval_only:
                # Include-only semantics: skip everything not in the allow list.
                if scene not in self.include_scenes:
                    n_not_in_include += 1
                    continue
            else:
                # Exclude held-out eval scenes
                if scene in self.exclude_scenes:
                    n_excluded += 1
                    continue
            scene_path = os.path.join(self.ROOT, scene)
            img_dir = os.path.join(scene_path, _SUBDIR_IMAGES)
            if not os.path.isdir(img_dir):
                continue
            basenames = sorted(
                f[: -len(_IMG_EXT)] for f in os.listdir(img_dir) if f.endswith(_IMG_EXT)
            )
            num_imgs = len(basenames)
            cut_off = self.num_views
            if num_imgs < cut_off:
                print(f"Skipping TartanGround {scene_path} (only {num_imgs} images)")
                continue

            env = self._scene_env(scene)
            env_counter[env] = env_counter.get(env, 0) + 1

            img_ids = list(np.arange(num_imgs) + offset)
            if self.max_starts_per_scene is not None:
                k = min(self.max_starts_per_scene, num_imgs)
                rel_starts = np.linspace(0, num_imgs - 1, k).astype(int)
                # ``linspace`` may produce duplicates when num_imgs < k; dedupe in order.
                _seen = set()
                rel_starts = [int(r) for r in rel_starts if not (r in _seen or _seen.add(r))]
                start_img_ids_ = [offset + r for r in rel_starts]
            else:
                start_img_ids_ = img_ids  # foldback: every frame is a valid start

            scenes.append(scene_path)
            scene_envs.append(env)
            scene_img_list.append(img_ids)
            sceneids.extend([scene_index] * num_imgs)
            images.extend(basenames)
            start_img_ids.extend(start_img_ids_)
            offset += num_imgs
            scene_index += 1

        self.scenes = scenes
        self.scene_envs = scene_envs
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list
        print(f"Loaded TartanGroundLong dataset done. {len(scenes)} scenes, {offset} frames.")
        # Summarize per-env coverage and which envs hit overrides
        n_overridden = sum(1 for e in env_counter if e in self.interval_by_env)
        n_total_overridden_scenes = sum(c for e, c in env_counter.items() if e in self.interval_by_env)
        print(f"  envs total: {len(env_counter)}  (with override: {n_overridden}, default: {len(env_counter) - n_overridden})")
        print(f"  scenes using override: {n_total_overridden_scenes} / {len(scenes)} = "
              f"{n_total_overridden_scenes / max(len(scenes), 1):.1%}")
        print(f"  default bounds (un-listed envs): ({self.min_interval}, {self.max_interval})")
        if self.eval_only:
            print(
                f"  EVAL-ONLY mode: include_scenes={len(self.include_scenes)} requested, "
                f"{n_not_in_include} on-disk scenes skipped (not in allow list), "
                f"max_starts_per_scene={self.max_starts_per_scene}, "
                f"total starts={len(start_img_ids)}"
            )
        else:
            print(f"  held-out for eval: {n_excluded} scenes (not seen during training)")

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def get_stats(self):
        return f"{len(self)} groups of views"

    def _get_views(
        self, idx, resolution, rng, num_views, preserve_fov, sequence_aug
    ):
        start_id = self.start_img_ids[idx]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]

        # Pick interval bounds for this scene's environment.
        env = self.scene_envs[scene_id]
        if env in self.interval_by_env:
            eff_min, eff_max = self.interval_by_env[env]
        else:
            eff_min, eff_max = self.min_interval, self.max_interval

        pos, ordered_video = foldback_from_start_long(
            num_views=num_views,
            id_ref=start_id,
            ids_all=all_image_ids,
            rng=rng,
            min_interval=int(eff_min),
            max_interval=int(eff_max),
            forward_only=self.forward_only,
            recent_stride_memory=self.recent_stride_memory,
            fix_interval_prob=self.fix_interval_prob,
        )
        image_idxs = np.array(all_image_ids)[pos]

        views = []
        for view_order, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = self.scenes[scene_id]
            basename = self.images[view_idx]

            img_path = osp.join(scene_dir, _SUBDIR_IMAGES, basename + _IMG_EXT)
            image = imread_cv2(img_path)
            depth_path = osp.join(scene_dir, _SUBDIR_DEPTHS, basename + _DEPTH_EXT)
            depthmap = np_load(depth_path)
            cam_path = osp.join(scene_dir, _SUBDIR_CAMERAS, basename + _CAM_EXT)
            camera_params = np_load(cam_path)

            intrinsics = camera_params["camera_intrinsics"]
            camera_pose = camera_params["camera_pose"]

            sky_mask = depthmap >= _SKY_DEPTH_THRESHOLD
            depthmap[sky_mask] = -1.0
            depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)
            # reading mask
            mask_path = osp.join(scene_dir, "masks", basename + _DEPTH_EXT)
            mask = np_load(mask_path).astype(bool)
            assert mask.shape == depthmap.shape, f"{mask.shape} vs {depthmap.shape}"
            depthmap[mask] = 0.0
            
            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image,
                depthmap,
                intrinsics,
                resolution,
                rng,
                preserve_fov=preserve_fov,
                sequence_aug=sequence_aug,
                info=(scene_dir, basename + _IMG_EXT),
            )

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=camera_pose, #c2w
                    camera_intrinsics=intrinsics,
                    dataset="tartanground",
                    label=scene_dir,
                    is_metric=self.is_metric,
                    instance=scene_dir + "_" + basename,
                    is_video=ordered_video,
                    quantile=np.array(1.0, dtype=np.float32),
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
