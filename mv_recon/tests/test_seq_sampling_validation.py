import json

from omegaconf import OmegaConf

from mv_recon.seq_sampling import ensure_seq_id_map, select_seq_id_map, validate_seq_id_map


class FakeDataset:
    sequence_list = ["a", "b"]
    sizes = {"a": 3, "b": 2}

    def get_seq_framenum(self, sequence_name):
        return self.sizes[sequence_name]


def test_valid_map_passes():
    cfg = OmegaConf.create({"strategy": "all"})
    valid, reason = validate_seq_id_map({"a": [0, 1, 2], "b": [0, 1]}, FakeDataset(), cfg)
    assert valid, reason


def test_stale_map_fails_closed_without_explicit_rebuild(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"a": [0, 1, 2, 3], "b": [0, 1]}))
    cfg = OmegaConf.create({"strategy": "all"})
    try:
        ensure_seq_id_map(str(path), FakeDataset(), cfg)
    except ValueError as exc:
        assert "Refusing to rebuild it implicitly" in str(exc)
    else:
        raise AssertionError("Expected an invalid existing map to fail closed")
    assert json.loads(path.read_text()) == {"a": [0, 1, 2, 3], "b": [0, 1]}


def test_stale_map_is_rebuilt_only_when_forced(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"a": [0, 1, 2, 3], "b": [0, 1]}))
    cfg = OmegaConf.create({"strategy": "all"})
    result = ensure_seq_id_map(str(path), FakeDataset(), cfg, force=True)
    assert result == {"a": [0, 1, 2], "b": [0, 1]}
    assert json.loads(path.read_text()) == result


def test_select_seq_id_map_preserves_requested_order_and_complete_sequences():
    source = {"a": [0, 1, 2], "b": [0, 1], "c": [0]}
    assert select_seq_id_map(source, ["c", "a"]) == {"c": [0], "a": [0, 1, 2]}


def test_select_seq_id_map_rejects_unknown_or_duplicate_names():
    source = {"a": [0], "b": [0]}
    for requested in (["missing"], ["a", "a"], []):
        try:
            select_seq_id_map(source, requested)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {requested}")
