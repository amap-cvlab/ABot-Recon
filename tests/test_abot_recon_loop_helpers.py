from types import SimpleNamespace

import torch
from omegaconf import DictConfig
from PIL import Image

from interfaces import abot_recon as abot_recon_interface


def test_loop_postprocess_returns_metadata_without_stream_forward(tmp_path, monkeypatch):
    salad = tmp_path / "salad.ckpt"
    dino = tmp_path / "dino.pth"
    salad.write_bytes(b"test")
    dino.write_bytes(b"test")
    base = torch.eye(4).repeat(3, 1, 1)
    calls = []

    def fake_apply(model, paths, poses, **kwargs):
        calls.append((model, paths, poses.clone(), kwargs))
        return poses.clone(), {"num_loop_edges": 2}

    monkeypatch.setattr("abot_recon.loop_closure.apply_loop_closure", fake_apply)
    model = SimpleNamespace(runtime_model=object())
    cfg = DictConfig(
        {
            "device": "cpu",
            "output_dir": str(tmp_path / "loop"),
            "abot_recon_loop_salad_ckpt": str(salad),
            "abot_recon_loop_dino_weights": str(dino),
        }
    )
    poses, metadata = abot_recon_interface.run_abot_recon_loop_from_c2w(
        ["a", "b", "c"], base, model, cfg, return_metadata=True
    )
    assert torch.equal(poses, base)
    assert metadata["num_loop_edges"] == 2
    assert len(calls) == 1


def test_descriptor_extraction_uses_images_only(tmp_path, monkeypatch):
    paths = []
    for index in range(2):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (640, 360), color=(index, 20, 30)).save(path)
        paths.append(str(path))
    submitted = []
    worker = SimpleNamespace(
        submit=lambda image: submitted.append(image),
        finish=lambda: ("descriptors", {"frames": len(submitted)}),
        cancel=lambda: None,
    )
    monkeypatch.setattr(
        "abot_recon.loop_closure.start_descriptor_worker", lambda **kwargs: worker
    )
    model = SimpleNamespace(height=280, width=504, fov_pad_rgb=(0.485, 0.456, 0.406))
    cfg = DictConfig(
        {
            "device": "cpu",
            "abot_recon_loop_salad_ckpt": "a",
            "abot_recon_loop_dino_weights": "b",
        }
    )
    result = abot_recon_interface.extract_abot_recon_loop_descriptors(paths, model, cfg)
    assert result == ("descriptors", {"frames": 2})
    assert [tuple(image.shape) for image in submitted] == [(1, 1, 3, 280, 504)] * 2
