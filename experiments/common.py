"""Shared helpers for experiment scripts."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"


class ConfigError(ValueError):
    pass


_REQUIRED = {
    "name": str, "cipher": str, "seed": int, "reference_difference": str,
    "pool": dict, "scoring": dict, "variability": dict, "neural": dict,
}


def validate_config(cfg: dict) -> dict:
    """Fail early, with a clear message, on missing keys or invalid values."""
    for k, typ in _REQUIRED.items():
        if k not in cfg:
            raise ConfigError(f"config missing key {k!r}")
        if not isinstance(cfg[k], typ):
            raise ConfigError(f"config key {k!r} must be {typ.__name__}")
    ciphers = {p.stem for p in (ROOT / "third_party/AutoND/ciphers").glob("*.py")}
    if ciphers and cfg["cipher"] not in ciphers:
        raise ConfigError(f"unknown cipher {cfg['cipher']!r}; AutoND provides {sorted(ciphers)}")
    int(cfg["reference_difference"], 16)
    p, sc, a = cfg["pool"], cfg["scoring"], cfg["scoring"].get("adaptive", {})
    checks = [
        (p.get("num_generations", 0) >= 1, "pool.num_generations >= 1"),
        (p.get("nb_samples", 0) >= 10, "pool.nb_samples >= 10"),
        (0 < p.get("threshold", 0) < 0.5, "0 < pool.threshold < 0.5"),
        (sc.get("k", 0) >= 1, "scoring.k >= 1"),
        (sc.get("t_fixed", 0) >= 10, "scoring.t_fixed >= 10"),
        (sc.get("t_reference", 0) > sc.get("t_fixed", 0), "scoring.t_reference > t_fixed"),
        (len(sc.get("seeds", [])) >= 1, "scoring.seeds non-empty"),
        (all(0 < f <= 1 for f in sc.get("budget_fractions", [1])), "0 < budget_fractions <= 1"),
        (0 < a.get("alpha", 0.01) < 0.5, "0 < adaptive.alpha < 0.5"),
        (a.get("growth", 2.0) > 1, "adaptive.growth > 1"),
        (0 < a.get("t0", 1) <= a.get("t_max", 1), "0 < adaptive.t0 <= adaptive.t_max"),
        (a.get("interval", "bias_bounded") in ("bias_bounded", "mcdiarmid", "basic", "percentile"),
         "adaptive.interval valid"),
        (cfg["neural"].get("start_round", 1) <= cfg["neural"].get("target_round", 1), "neural start_round <= target_round"),
    ]
    bad = [msg for ok, msg in checks if not ok]
    if bad:
        raise ConfigError("invalid config: " + "; ".join(bad))
    return cfg


def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    try:
        cfg = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ConfigError(f"config {p} is not valid JSON: {e}") from e
    validate_config(cfg)
    cfg["_config_path"] = str(p.resolve())
    return cfg


def out_dir(*parts) -> Path:
    p = RESULTS.joinpath(*map(str, parts))
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_default))


def _default(o):
    import numpy as np
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.seconds = time.perf_counter() - self.t0
