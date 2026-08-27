"""Resolve per-model output directory names for relpose eval scripts."""

from __future__ import annotations

import os
import re
from typing import Any, Optional


def sanitize_output_slug(name: str, max_len: int = 220) -> str:
    s = re.sub(r"[^\w.\-]+", "_", str(name).strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if s else "model"


def ckpt_tag_from_path(ckpt: Optional[str]) -> str:
    if not ckpt:
        return ""
    m = re.search(r"checkpoint_(\d+)", str(ckpt))
    if m:
        return f"ckpt{m.group(1)}"
    base = os.path.splitext(os.path.basename(str(ckpt)))[0]
    return sanitize_output_slug(base, max_len=32)



def resolve_model_output_slug(
    model_keyname: str,
    model_info: Any,
    model: Optional[Any] = None,
) -> str:
    """
    Directory name under ``output_dir`` for this model run.

    Priority: ``model_info.eval_output_name`` > model-provided auto slug > ``model_keyname``.
    """
    explicit = model_info.get("eval_output_name", None) if model_info is not None else None
    if explicit is not None and str(explicit).strip():
        return sanitize_output_slug(str(explicit).strip())

    if model is not None:
        slug_fn = getattr(model, "eval_output_slug", None)
        if callable(slug_fn):
            return sanitize_output_slug(slug_fn(model_keyname))

    return sanitize_output_slug(model_keyname)
