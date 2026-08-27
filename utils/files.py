import json
import os
import os.path as osp
import glob
import re
from typing import Optional
from omegaconf import DictConfig, ListConfig

# Sentinel for JSON-driven sequence/image lists (e.g. Oxford Spires).
_FROM_JSON_TOKEN = "__from_json__"

# Cache loaded JSON summaries so we don't re-parse the file per sequence.
_JSON_CACHE: dict = {}

def _load_json_summary(json_file: str) -> dict:
    if json_file not in _JSON_CACHE:
        with open(json_file, "r", encoding="utf-8") as f:
            _JSON_CACHE[json_file] = json.load(f)
    return _JSON_CACHE[json_file]

def get_all_sequences(dataset_cfg: DictConfig, sort_by_seq_name: bool = True):
    ls_all = dataset_cfg.ls_all_seqs

    if isinstance(ls_all, str) and ls_all == _FROM_JSON_TOKEN:
        # JSON-driven: take all "selected" scenes from json_file
        json_file = dataset_cfg.json_file
        summary = _load_json_summary(json_file)
        seq_list = [
            scene for scene, entry in summary["scenes"].items()
            if entry.get("selected", False)
        ]
    elif isinstance(ls_all, str):
        # ls_all_seqs is a directory path of sequence subdirs
        seq_list = [d for d in os.listdir(ls_all) if osp.isdir(osp.join(ls_all, d))]
    elif isinstance(ls_all, ListConfig):
        seq_list = list(ls_all)
    else:
        raise ValueError(
            f"Unknown ls_all_seqs type: {type(ls_all)}, ls_all_seqs is {ls_all}, "
            f"which should be a string, a ListConfig, or '{_FROM_JSON_TOKEN}'"
        )
    return sorted(seq_list) if sort_by_seq_name else seq_list

def _natural_key(path: str):
    """Return a key for natural sorting of filenames.

    Splits the basename by digit groups and converts digit parts to integers
    so that filenames like '2.png' < '10.png'.
    """
    name = osp.basename(path)
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

def list_imgs_a_sequence(dataset_cfg: DictConfig, seq: Optional[str] = None):
    img_cfg = dataset_cfg.img

    # JSON-driven image list (e.g. Oxford Spires: only the 3840 selected frames)
    if img_cfg.get("source", None) == "json":
        json_file = dataset_cfg.json_file
        summary = _load_json_summary(json_file)
        entry = summary["scenes"][seq]
        if not entry.get("selected", False):
            return []
        img_dir = img_cfg.path.format(seq=seq)
        return [osp.join(img_dir, fname) for fname in entry["image_files"]]

    subdir = img_cfg.path.format(seq=seq)
    ext = img_cfg.ext
    filelist = sorted(glob.glob(f"{subdir}/*.{ext}"), key=_natural_key)
    return filelist