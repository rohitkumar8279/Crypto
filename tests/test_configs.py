import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from common import ConfigError, load_config, validate_config  # noqa: E402


@pytest.mark.parametrize("name", ["simon32_demo", "simon32_main"])
def test_shipped_configs_are_valid(name):
    cfg = load_config(ROOT / "experiments/configs" / f"{name}.json")
    assert cfg["name"] == name


def test_invalid_configs_rejected(tmp_path):
    base = json.loads((ROOT / "experiments/configs/simon32_demo.json").read_text())
    bad = copy.deepcopy(base); del bad["pool"]
    with pytest.raises(ConfigError):
        validate_config(bad)
    bad = copy.deepcopy(base); bad["cipher"] = "gift64"
    with pytest.raises(ConfigError):
        validate_config(bad)
    bad = copy.deepcopy(base); bad["scoring"]["adaptive"]["alpha"] = 2
    with pytest.raises(ConfigError):
        validate_config(bad)
    f = tmp_path / "broken.json"; f.write_text("{not json")
    with pytest.raises(ConfigError):
        load_config(f)
