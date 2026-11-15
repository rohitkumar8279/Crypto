"""Import AutoND modules from the pinned, unmodified checkout."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTOND_DIR = PROJECT_ROOT / "third_party" / "AutoND"
AUTOND_REPO = "https://github.com/Crypto-TII/AutoND"
# Commit audited in Phase 1 (2025-03-13). Change only together with a new audit.
AUTOND_PINNED_COMMIT = "8d66e0a43a9e6fbbb3624d68296c1b77aa2da779"


def ensure_on_path() -> None:
    # Never write __pycache__ into the pinned third-party checkout (keeps `git status` clean).
    sys.dont_write_bytecode = True
    if not AUTOND_DIR.is_dir():
        raise FileNotFoundError(
            f"AutoND checkout not found at {AUTOND_DIR}. Run scripts/setup_autond.sh first."
        )
    p = str(AUTOND_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_cipher(name: str):
    """Return AutoND's cipher module (e.g. 'simon3264').

    Importing ``ciphers.simon3264`` runs AutoND's own test-vector assertion.
    """
    ensure_on_path()
    return importlib.import_module(f"ciphers.{name}")


def load_optimizer():
    """AutoND's optimizer module (NumPy only; no TensorFlow import)."""
    ensure_on_path()
    return importlib.import_module("optimizer")


def load_dbitnet():
    """AutoND's DBitNet module (TensorFlow is imported lazily inside make_model)."""
    ensure_on_path()
    return importlib.import_module("dbitnet")


def autond_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(AUTOND_DIR), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def autond_is_clean() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(AUTOND_DIR), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() == ""
    except Exception:
        return None


def check_pinned(strict: bool = True) -> str | None:
    c = autond_commit()
    if c != AUTOND_PINNED_COMMIT:
        msg = f"AutoND commit {c} != pinned {AUTOND_PINNED_COMMIT}"
        if strict:
            raise RuntimeError(msg)
        print("WARNING:", msg, file=sys.stderr)
    return c
