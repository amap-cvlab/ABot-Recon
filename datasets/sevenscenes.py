import os.path as osp
import os 
import numpy as np
import torch
import imageio.v2
import torchvision.transforms as tvf

from typing import Optional, Union, List
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm
import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from models.vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from datasets.utils.cropping import resize_image, resize_image_depth_and_intrinsic

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
to_tensor = tvf.ToTensor()

class SevenScenes(Dataset):
    def __init__(
        self,
        SEVENSCENES_DIR: str,
        split: str = 'test',
        load_img_size: int = 518,
        cache_file: str = "data/dataset_cache/7scenes_mv_recon_cache.npy",
    ):
        
        self.SEVENSCENES_DIR = SEVENSCENES_DIR
        if SEVENSCENES_DIR == None:
            raise NotImplementedError
        print(f"SEVENSCENES_DIR is {SEVENSCENES_DIR}")

        self.split = split
        # train/test: official Split.txt; seq01: Horizon-paper split (one seq-01 per scene).
        if self.split in ['train', 'test']:
            split_txt_name = f"{self.split.capitalize()}Split.txt"
        elif self.split in ['seq01', 'hs-paper', 'horizon_seq01']:
            split_txt_name = None
        else:
            raise ValueError(
                f"Invalid split: {self.split}. Must be 'train', 'test', or 'seq01'."
            )

        if osp.exists(cache_file):
            print(f"[7Scenes] Loading from cache file: {cache_file}")
            self.metadata = np.load(cache_file, allow_pickle=True).item()
            self.sequence_list = sorted(list(self.metadata.keys()))
        else:
            print(f"[7Scenes] Cache file not found, loading from {SEVENSCENES_DIR}")

            scene_list = sorted(
                [d for d in os.listdir(SEVENSCENES_DIR) if osp.isdir(osp.join(SEVENSCENES_DIR, d))]
            )

            self.sequence_list = []
            if split_txt_name is None:
                # Horizon Table-4 style: every scene's seq-01 (may be train or test).
                for scene in scene_list:
                    seq = f"{scene}/seq-01"
                    if osp.isdir(osp.join(SEVENSCENES_DIR, seq)):
                        self.sequence_list.append(seq)
                if len(self.sequence_list) != 7:
                    raise ValueError(
                        f"Expected 7 seq-01 sequences, found {len(self.sequence_list)}: "
                        f"{self.sequence_list}"
                    )
            else:
                for scene in scene_list:
                    scene_dir = osp.join(SEVENSCENES_DIR, scene)
                    split_txt_file = osp.join(scene_dir, split_txt_name)
                    # from "sequenceX" to "scene/seq-0X"
                    seq_name_trans = lambda rawname: f"{scene}/seq-{int(rawname[len('sequence'):]):02d}"
                    with open(split_txt_file, 'r') as f:
                        scene_seq_list = [seq_name_trans(line.strip()) for line in f.readlines()]
                    self.sequence_list.extend(scene_seq_list)
            
            self.metadata = {}
            for seq in tqdm(self.sequence_list):
                seq_path = osp.join(SEVENSCENES_DIR, seq)
                
                list_seqs_func = lambda suffix: sorted(
                    [d for d in os.listdir(seq_path) if d.endswith(suffix)]
                )
                seq_rgbs = list_seqs_func('.color.png')
                seq_depths = list_seqs_func('.depth.proj.png')
                seq_poses = list_seqs_func('.pose.txt')

                if not (len(seq_rgbs) == len(seq_depths) == len(seq_poses)):
                    raise ValueError(f"Sequence {seq} has mismatched number of RGB, depth, and pose files: {len(seq_rgbs)}, {len(seq_depths)}, {len(seq_poses)}")
                get_number_func = lambda x: int(x.split('-')[1].split('.')[0])
                if not (get_number_func(seq_rgbs[0]) == get_number_func(seq_depths[0]) == get_number_func(seq_poses[0]) == 0):
                    raise ValueError(f"First frame of sequence {seq} has number of RGB, depth, and pose != 0: {seq_rgbs[0]}, {seq_depths[0]}, {seq_poses[0]}")
                if not (get_number_func(seq_rgbs[-1]) == get_number_func(seq_depths[-1]) == get_number_func(seq_poses[-1]) == len(seq_rgbs) - 1):
                    raise ValueError(f"Last frame of sequence {seq} has number of RGB, depth, and pose != N-1: {seq_rgbs[-1]}, {seq_depths[-1]}, {seq_poses[-1]}, N={len(seq_rgbs)}")
                
                self.metadata[seq] = len(seq_rgbs)

            os.makedirs(osp.dirname(cache_file), exist_ok=True)
            np.save(cache_file, self.metadata)

        self.load_img_size = load_img_size
        print(f"[7Scenes] Data size: {len(self)}")

    def __len__(self):
        return len(self.sequence_list)

    def get_seq_framenum(self, index: Optional[int] = None, sequence_name: Optional[str] = None):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]
        return self.metadata[sequence_name]

    def __getitem__(self, idx_N):
        """Fetch item by index and a dynamic variable n_per_seq."""

        # Different from most pytorch datasets,
        # here we not only get index, but also a dynamic variable n_per_seq
        # supported by DynamicBatchSampler

        index, n_per_seq = idx_N
        sequence_name = self.sequence_list[index]
        metadata = self.metadata[sequence_name]
        ids = np.random.choice(len(metadata), n_per_seq, replace=False)
        return self.get_data(index=index, ids=ids)

    def get_data(
            self,
            index: Optional[int] = None,
            sequence_name: Optional[str] = None,
            ids: Union[List[int], np.ndarray, None] = None,
        ):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name: str = self.sequence_list[index]
        seq_len: int = self.metadata[sequence_name]

        if ids is None:
            ids = np.arange(seq_len).tolist()
        elif isinstance(ids, np.ndarray):
            assert ids.ndim == 1, f"ids should be a 1D array, but got {ids.ndim}D"
            ids = ids.tolist()

        fx, fy, cx, cy = 525, 525, 320, 240  # hard code

        image_paths: list      = [""] * len(ids)
        images: list           = [0]  * len(ids)
        depths: list           = [0] * len(ids)
        extrinsics: np.ndarray = np.zeros((len(ids), 3, 4))  # (S, 3, 4)
        intrinsics: np.ndarray = np.tile(
            np.array(
                [
                    [fx, 0,  cx],
                    [0,  fy, cy],
                    [0,  0,  1 ]
                ],
                dtype=np.float32
            ),
            reps=(len(ids), 1, 1)
        )  # (S, 3, 3)

        for id_index, id in enumerate(ids):
            
            impath: str    = osp.join(self.SEVENSCENES_DIR, sequence_name, f"frame-{id:06d}.color.png")
            depthpath: str = osp.join(self.SEVENSCENES_DIR, sequence_name, f"frame-{id:06d}.depth.proj.png")
            posepath: str  = osp.join(self.SEVENSCENES_DIR, sequence_name, f"frame-{id:06d}.pose.txt")

            rgb_image: Image.Image = Image.open(impath)
            depthmap: np.ndarray   = imageio.v2.imread(depthpath)
            assert depthmap.shape == (480, 640), f"Depth map shape {depthmap.shape} does not match expected (480, 640)"
            rgb_image: Image.Image = resize_image(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            depthmap[depthmap == 65535] = 0
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0
            depthmap[depthmap > 10] = 0    # to far, invalid
            depthmap[depthmap < 1e-3] = 0  # to near, invalid

            # load_img_size <= 0: keep native GT resolution (for nearest_depth_to_gt).
            if self.load_img_size is not None and int(self.load_img_size) > 0:
                rgb_image, depthmap, intrinsics[id_index] = resize_image_depth_and_intrinsic(
                    image=rgb_image,
                    depth_map=depthmap,
                    intrinsic=intrinsics[id_index],
                    output_width=self.load_img_size,
                )

            camera_pose = np.loadtxt(posepath)

            image_paths[id_index] = impath
            images[id_index]      = to_tensor(rgb_image)
            depths[id_index]      = depthmap
            extrinsics[id_index]  = closed_form_inverse_se3(camera_pose[None])[0][:3, :]

        depths = np.array(depths)  # (S, H, W)
        pointclouds = unproject_depth_map_to_point_map(
            depth_map=depths[..., None],
            intrinsics_cam=intrinsics,
            extrinsics_cam=extrinsics
        )

        # Build full (N, 4, 4) w2c extrinsics for relpose evaluation
        extrs_4x4 = np.zeros((len(ids), 4, 4), dtype=np.float32)
        extrs_4x4[:, :3, :] = extrinsics
        extrs_4x4[:, 3, 3] = 1.0

        batch = {"seq_id": sequence_name, "seq_len": seq_len, "ind": torch.tensor(ids)}
        batch['image_paths'] = image_paths  # list of str
        batch['images']      = torch.stack(images, dim=0)
        batch['pointclouds'] = pointclouds  # in numpy
        batch['valid_mask']  = depths > 1e-4
        batch["extrs"] = torch.from_numpy(extrs_4x4)   # (N, 4, 4) w2c
        batch["intrs"] = torch.from_numpy(intrinsics)   # (N, 3, 3)

        return batch
