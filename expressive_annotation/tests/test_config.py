"""Configuration defaults and operator overrides."""
from __future__ import annotations

import pytest

from continuo_expressive import config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("HF_ENDPOINT", "CONTINUO_EXPRESSIVE_TRUST_REMOTE_CODE",
                 "CONTINUO_EXPRESSIVE_VOXLECT_REPO", "CONTINUO_EXPRESSIVE_CAPTIONER_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_hf_endpoint_is_unset_by_default(monkeypatch):
    assert config.apply_hf_endpoint() is None
    assert "HF_ENDPOINT" not in __import__("os").environ


def test_an_explicit_endpoint_is_used(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://example.org/hf")
    assert config.apply_hf_endpoint() == "https://example.org/hf"


def test_trust_remote_code_is_off_by_default():
    assert config.trust_remote_code() is False


def test_trust_remote_code_needs_an_explicit_truthy_value(monkeypatch):
    monkeypatch.setenv("CONTINUO_EXPRESSIVE_TRUST_REMOTE_CODE", "0")
    assert config.trust_remote_code() is False
    monkeypatch.setenv("CONTINUO_EXPRESSIVE_TRUST_REMOTE_CODE", "maybe")
    assert config.trust_remote_code() is False
    monkeypatch.setenv("CONTINUO_EXPRESSIVE_TRUST_REMOTE_CODE", "yes")
    assert config.trust_remote_code() is True


def test_paths_default_under_the_repo_and_are_overridable(monkeypatch):
    assert config.voxlect_repo().parent == config.ROOT / "third_party"
    monkeypatch.setenv("CONTINUO_EXPRESSIVE_VOXLECT_REPO", "models/voxlect")
    assert str(config.voxlect_repo()) == "models/voxlect"


def test_model_ids_are_overridable(monkeypatch):
    assert config.captioner_model().startswith("Qwen/")
    monkeypatch.setenv("CONTINUO_EXPRESSIVE_CAPTIONER_MODEL", "models/captioner")
    assert config.captioner_model() == "models/captioner"


def test_no_absolute_machine_paths_are_baked_in():
    # every default must be repo-relative; a machine-specific path would
    # make the checkout run on exactly one machine
    for resolver in (config.voxprofile_repo, config.voxlect_repo,
                     config.firered_repo, config.firered_model_dir):
        assert config.ROOT in resolver().parents
