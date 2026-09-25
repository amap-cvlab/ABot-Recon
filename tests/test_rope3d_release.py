from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_internal_rope_uses_release_names():
    source_root = ROOT / "abot_recon"
    inherited_prefix = "ling" + "bot"
    forbidden = (
        inherited_prefix + "_3d",
        inherited_prefix + "_rope",
        inherited_prefix + "_wan_rope",
        "rope." + inherited_prefix,
    )
    offenders = []
    for path in source_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in forbidden):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []

    model_source = (source_root / "model.py").read_text(encoding="utf-8")
    assert 'global_pos_encoding="rope3d"' in model_source
    assert "rope3d_config={" in model_source


def test_curope_build_sources_are_shipped():
    curope = ROOT / "abot_recon/modeling/pi3/models/curope"
    for name in ("setup.py", "curope.cpp", "kernels.cu"):
        assert (curope / name).is_file()

    position_source = (ROOT / "abot_recon/modeling/pi3/models/layers/pos_embed.py").read_text()
    assert "from ..curope import cuRoPE2D" in position_source
    assert "from models.curope" not in position_source
