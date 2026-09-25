"""Causal streaming network used by the released ABot-Recon model."""

from typing import TYPE_CHECKING

__all__ = ["ABotReconNetwork"]

if TYPE_CHECKING:
    from abot_recon.modeling.streaming.network import ABotReconNetwork


def __getattr__(name: str):
    if name == "ABotReconNetwork":
        from abot_recon.modeling.streaming.network import ABotReconNetwork

        return ABotReconNetwork
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
