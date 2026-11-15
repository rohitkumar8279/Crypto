"""Reproducibility manifest written next to every experiment output."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

from .autond_bridge import PROJECT_ROOT, autond_commit, autond_is_clean, AUTOND_PINNED_COMMIT

PACKAGES = ["numpy", "scipy", "pandas", "matplotlib", "tensorflow", "tensorflow-intel", "keras"]


def source_hash() -> str:
    """SHA-256 over this project's own Python sources (cgsearch/, experiments/)."""
    h = hashlib.sha256()
    for sub in ("cgsearch", "experiments"):
        for p in sorted((PROJECT_ROOT / sub).rglob("*.py")):
            h.update(str(p.relative_to(PROJECT_ROOT)).replace("\\", "/").encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def package_versions() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


def hardware() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "cpu_count": os.cpu_count()}
    if "tensorflow" in sys.modules:  # never import TF just for the manifest
        tf = sys.modules["tensorflow"]
        try:
            info["tf_devices"] = [d.name for d in tf.config.list_physical_devices()]
            gpus = tf.config.list_physical_devices("GPU")
            info["gpu_details"] = [tf.config.experimental.get_device_details(g) for g in gpus]
        except Exception as e:  # pragma: no cover
            info["tf_devices_error"] = repr(e)
    return info


def build_manifest(config: dict, extra: dict | None = None) -> dict:
    return {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "argv": sys.argv,
        "python": sys.version,
        "packages": package_versions(),
        "hardware": hardware(),
        "autond_commit": autond_commit(),
        "autond_pinned_commit": AUTOND_PINNED_COMMIT,
        "autond_worktree_clean": autond_is_clean(),
        "project_source_sha256": source_hash(),
        "config": config,
        **(extra or {}),
    }


def write_manifest(out_dir: str | Path, config: dict, extra: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "manifest.json"
    p.write_text(json.dumps(build_manifest(config, extra), indent=2, default=str))
    return p
