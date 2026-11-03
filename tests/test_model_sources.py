"""Immutable snapshot loading does not depend on an existing refs/main cache."""
import sys
from types import SimpleNamespace

import pytest

from proteocraft.model_sources import MODEL_REVISIONS, resolve_model_source


@pytest.mark.parametrize("name", list(MODEL_REVISIONS))
def test_official_model_uses_exact_cached_snapshot(name, monkeypatch):
    calls = []
    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return "/cache/snapshots/" + kwargs["revision"]
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    assert resolve_model_source(name) == "/cache/snapshots/" + MODEL_REVISIONS[name]
    assert calls == [{"repo_id": name, "revision": MODEL_REVISIONS[name], "local_files_only": True}]


@pytest.mark.parametrize("name", ["/models/custom-snapshot", "other/model"])
def test_explicit_custom_model_location_is_preserved(name, monkeypatch):
    def unexpected(**kwargs):
        pytest.fail("Custom model location must not be redirected to the official snapshot")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=unexpected))
    assert resolve_model_source(name) == name


def test_missing_pinned_snapshot_is_not_silently_replaced(monkeypatch):
    def missing(**kwargs):
        raise FileNotFoundError("Pinned snapshot absent")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=missing))
    with pytest.raises(FileNotFoundError, match="Pinned snapshot absent"):
        resolve_model_source("biohub/ESMFold2")
