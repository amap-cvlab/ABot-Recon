from __future__ import annotations

import pandas as pd

from utils.messages import write_csv


def test_write_csv_keeps_append_behavior_without_keys(tmp_path):
    path = tmp_path / "metrics.csv"
    write_csv(str(path), {"seq": "00", "ATE": 1.0})
    write_csv(str(path), {"seq": "00", "ATE": 2.0})

    result = pd.read_csv(path)
    assert result.to_dict("records") == [
        {"seq": 0, "ATE": 1.0},
        {"seq": 0, "ATE": 2.0},
    ]


def test_write_csv_upserts_matching_logical_row(tmp_path):
    path = tmp_path / "metrics.csv"
    keys = ("model", "dataset", "seq")
    write_csv(
        str(path),
        {"model": "ours", "dataset": "kitti", "seq": "00", "ATE": 1.0},
        key_fields=keys,
    )
    write_csv(
        str(path),
        {"model": "ours", "dataset": "kitti", "seq": "01", "ATE": 2.0},
        key_fields=keys,
    )
    write_csv(
        str(path),
        {"model": "ours", "dataset": "kitti", "seq": "00", "ATE": 3.0},
        key_fields=keys,
    )


    result = pd.read_csv(path)
    assert len(result) == 2
    values = {int(row.seq): row.ATE for row in result.itertuples()}
    assert values == {0: 3.0, 1: 2.0}
