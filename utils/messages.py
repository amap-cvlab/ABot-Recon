import os
import os.path as osp
import sys
import pandas as pd

from typing import List

def set_default_arg(key: str, default_value: str):
    """
    check if `key` in arguments, else append default value `key=value` to argument list
    """
    has_key = any(arg.startswith(f"{key}=") for arg in sys.argv)
    if not has_key:
        sys.argv.append(f"{key}={default_value}")

def make_csvsdir_and_remove_history_csvs(input_root: str, seqs_csv_file: str):
    """
    Make the input directory for CSV files and remove any existing history CSV files.
    """
    if osp.isfile(seqs_csv_file):
        os.remove(seqs_csv_file)
    os.makedirs(input_root, exist_ok=True)
    for file in os.listdir(input_root):
        if file.endswith(".csv"):
            os.remove(osp.join(input_root, file))

def gather_csv_and_write(input_root: str, output_file: str):
    """
    Gather all CSV files in the input directory, concatenate them, and write to the output file.
    If the input directory contains multiple rows in a CSV file, only the last row will be saved.
    If the output file already exists, it will be overwritten.
    """
    seq_dfs = []
    for seq_csv_file in sorted(os.listdir(input_root)):
        if seq_csv_file.endswith(".csv"):
            df = pd.read_csv(osp.join(input_root, seq_csv_file))
            if len(df) > 1:
                print(f"Warning: {osp.join(input_root, seq_csv_file)} has more than one row, only the last row will be saved.")
                df = df.tail(1)
            seq_dfs.append(df)

    if len(seq_dfs) == 0:
        raise ValueError(f"No CSV files found in {input_root}. Returning an empty DataFrame.")

    df = pd.concat(seq_dfs, ignore_index=True)
    if osp.isfile(output_file):
        print(f"Warning: {output_file} already exists, data will be overwritten.")
    df.to_csv(output_file, index=False)
    return df

def model_checkpoint_path(cfg) -> str:
    """Resolve checkpoint path from model cfg (ABot-Recon uses ``ckpt``, others use ``pretrained_model_name_or_path``)."""
    for key in ("pretrained_model_name_or_path", "ckpt", "checkpoint"):
        if hasattr(cfg, key):
            val = getattr(cfg, key)
            if val is not None and str(val).strip():
                return str(val)
    return "???"


def write_csv(file_path: str, data_dict: dict, key_fields=None):
    # transform data of one row to DataFrame
    new_row = pd.DataFrame([data_dict])
    
    # Directly save when the file does not exist. With key_fields, replace an
    # existing logical row so resumed evaluations remain idempotent.
    if not osp.isfile(file_path):
        new_row.to_csv(file_path, index=False)
    else:
        existing_data = pd.read_csv(file_path)
        if key_fields:
            missing = [key for key in key_fields if key not in data_dict]
            if missing:
                raise KeyError(f"Missing CSV upsert keys: {missing}")
            matches = pd.Series(True, index=existing_data.index)

            def canonical_key(value):
                text = str(value)
                try:
                    numeric = float(text)
                except ValueError:
                    return text
                return str(int(numeric)) if numeric.is_integer() else text

            for key in key_fields:
                if key not in existing_data.columns:
                    matches[:] = False
                    break
                expected = canonical_key(data_dict[key])
                matches &= existing_data[key].map(canonical_key) == expected
            existing_data = existing_data.loc[~matches]
        updated_data = pd.concat([existing_data, new_row], ignore_index=True)
        temporary = f"{file_path}.tmp-{os.getpid()}"
        updated_data.to_csv(temporary, index=False)
        os.replace(temporary, file_path)

def format_matrix_str(matrix):
    def format_float(num, total_width=20, decimal_places=12):
        # strip all right 0s, then strip '.'
        s = f"{num:{total_width}.{decimal_places}f}".rstrip("0").rstrip(".")
        # add space to the left
        return f"{s:>{total_width}}"
    formatted = [
        [
            # f"{num:20.12f}".rstrip("0").rstrip(".") if "." in f"{num}" else f"{num:20}" 
            # f"{num:20.12f}" if "." in f"{num}" else f"{num:20}" 
            format_float(num, total_width=15, decimal_places=8)
            for num in row
        ]
        for row in matrix
    ]
    rows = [
        f"        [{', '.join(num for num in row)}]"
        for row in formatted
    ]
    return "    [\n" + ",\n".join(rows) + "\n    ]"

def save_list_of_matrices(matrices_tosave: List[List[List[float]]], save_path: str) -> None:
    json_str = "[\n" + ",\n".join(format_matrix_str(mat) for mat in matrices_tosave) + "\n]"
    with open(save_path, "w") as f:
        f.write(json_str)