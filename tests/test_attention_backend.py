import pytest

import abot_recon.model as model_module


def test_auto_prefers_paged(monkeypatch):
    monkeypatch.setattr(model_module, "flashinfer_available", lambda: True)
    assert model_module.resolve_attention_backend("auto") == "paged"


def test_auto_falls_back_to_sdpa(monkeypatch):
    monkeypatch.setattr(model_module, "flashinfer_available", lambda: False)
    assert model_module.resolve_attention_backend("auto") == "sdpa"


def test_explicit_paged_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(model_module, "flashinfer_available", lambda: False)
    with pytest.raises(RuntimeError, match="FlashInfer"):
        model_module.resolve_attention_backend("paged")


def test_explicit_sdpa_ignores_flashinfer(monkeypatch):
    monkeypatch.setattr(model_module, "flashinfer_available", lambda: True)
    assert model_module.resolve_attention_backend("sdpa") == "sdpa"
