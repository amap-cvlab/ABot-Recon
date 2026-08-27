import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mv_recon.runtime_manifest import (
    record_model_runtime,
    require_model_runtime,
    write_runtime_manifest,
)


class FakeModel:
    pass


def test_runtime_manifest_is_complete_and_idempotent(tmp_path, monkeypatch):
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (518, 388)).save(image_path)
    model = FakeModel()
    record_model_runtime(
        model,
        input_hw=(378, 518),
        input_storage_dtype="float32",
        forward_compute_dtype="bf16",
        preprocess="official",
        online_state="streaming",
        forward_frames=1,
    )
    monkeypatch.setattr(
        "mv_recon.runtime_manifest.cuda_runtime_name", lambda device: "test-device"
    )
    kwargs = dict(
        output_root=str(tmp_path / "output"),
        model_name="model",
        dataset_name="Oxford",
        sequence_name="seq/01",
        task="pointcloud",
        filelist=[str(image_path)],
        runtime=require_model_runtime(model),
        metric_frame_ids=np.array([0]),
        metric_frame_count=1,
        checkpoint="checkpoint.bin",
        device="cpu",
        protocol={"alignment": "sim3", "source_frame_ids": [0]},
    )
    csv_path, json_path = write_runtime_manifest(**kwargs)
    write_runtime_manifest(**kwargs)

    payload = json.loads(Path(json_path).read_text())
    assert (payload["source_rgb_h"], payload["source_rgb_w"]) == (388, 518)
    assert (payload["input_h"], payload["input_w"]) == (378, 518)
    assert payload["forward_compute_dtype"] == "bf16"
    assert payload["metric_frame_ids"] == [0]
    with Path(csv_path).open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1


def test_runtime_manifest_rejects_forward_frame_mismatch(tmp_path):
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (8, 6)).save(image_path)
    with pytest.raises(RuntimeError, match="forward-frame mismatch"):
        write_runtime_manifest(
            output_root=str(tmp_path / "output"),
            model_name="model",
            dataset_name="dataset",
            sequence_name="sequence",
            task="pointcloud",
            filelist=[str(image_path)],
            runtime={
                "input_h": 6,
                "input_w": 8,
                "input_storage_dtype": "float32",
                "forward_compute_dtype": "fp32",
                "preprocess": "none",
                "online_state": "streaming",
                "forward_frames": 2,
            },
            metric_frame_ids=[0],
            metric_frame_count=1,
            checkpoint="checkpoint.bin",
            device="cpu",
            protocol={},
        )
