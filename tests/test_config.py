import sys
from types import SimpleNamespace
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from flsim.config import build_config


def test_build_config_defaults(tmp_path: Path):
    args = SimpleNamespace()
    cfg = build_config(args, tmp_path)
    assert cfg.data.name == "mnist"
    assert cfg.train.nodes == 8
    assert cfg.paths.run_dir == tmp_path


def test_build_config_overrides(tmp_path: Path):
    args = SimpleNamespace(dataset="cifar10", nodes=5, agg_strategy="trimmed_mean")
    cfg = build_config(args, tmp_path)
    assert cfg.data.name == "cifar10"
    assert cfg.train.nodes == 5
    assert cfg.aggregation.strategy == "trimmed_mean"
