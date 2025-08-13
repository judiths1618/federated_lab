from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

@dataclass
class DataCfg:
    # NEW: dataset name for evaluator/loader selection
    name: str = "mnist"               # mnist | cifar10 | custom
    partitioner: str = "dirichlet"    # iid|dirichlet|pathological|shards
    alpha: float = 0.5
    classes_per_node: int = 2
    shards_per_node: int = 2
    samples_per_client: int = 0

@dataclass
class ModelCfg:
    # NEW: model hint for evaluator model selection
    name: Optional[str] = None        # e.g., "linear_mnist", "cifar_simple", "resnet18", ...

@dataclass
class TrainCfg:
    nodes: int = 8
    rounds: int = 5
    epochs: int = 1
    batch_size: int = 64
    lr: float = 0.05
    target_acc: float = 0.0
    upload_delta: bool = False
    save_updates: bool = False

@dataclass
class RewardCfg:
    base_reward: float = 100.0
    stake_weight: float = 0.4
    committee_size: int = 10
    hist_decay: float = 0.9
    reward_rate: float = 1.0
    penalize_negative: bool = False
    # NEW: used by contract incentives & detection
    committee_bonus: float = 0.10
    mal_eval_diff_thresh: float = 0.15

@dataclass
class AttackCfg:
    malicious_frac: float = 0.0        # 0.05 / 0.10 / ...
    malicious_strategy: str = "none"   # none|signflip|scaling|gaussian|label_flip|metric_spoof
    scale: float = 10.0                # for scaling/signflip (magnitude)
    noise_std: float = 0.1             # gaussian noise std
    spoof_mode: str = "none"           # none|high_acc|low_acc|random
    seed: int = 42

@dataclass
class PathsCfg:
    run_dir: Path = Path("./runs/tmp")
    models_dir: Path = Path("./runs/tmp/models")
    updates_dir: Path = Path("./runs/tmp/updates")
    sim_dir: Path = Path("./runs/tmp/simulation")
    log_csv: Path = Path("./runs/tmp/fl_log.csv")

@dataclass
class Cfg:
    data: DataCfg
    model: ModelCfg            # NEW
    train: TrainCfg
    reward: RewardCfg
    attack: AttackCfg
    paths: PathsCfg

def build_config(args, run_dir: Path):
    paths = PathsCfg(
        run_dir=run_dir,
        models_dir=run_dir / "models",
        updates_dir=run_dir / "updates",
        sim_dir=run_dir / "simulation",
        log_csv=run_dir / "fl_log.csv",
    )

    data = DataCfg()
    model = ModelCfg()
    train = TrainCfg()
    reward = RewardCfg()
    attack = AttackCfg()

    # ---------- overrides: data ----------
    if getattr(args, "dataset", None) is not None:    # NEW: --dataset
        data.name = args.dataset
    if getattr(args, "partitioner", None) is not None:
        data.partitioner = args.partitioner
    if getattr(args, "alpha", None) is not None:
        data.alpha = args.alpha
    if getattr(args, "classes_per_node", None) is not None:
        data.classes_per_node = args.classes_per_node
    if getattr(args, "shards_per_node", None) is not None:
        data.shards_per_node = args.shards_per_node
    if getattr(args, "samples_per_client", None) is not None:
        data.samples_per_client = args.samples_per_client

    # ---------- overrides: model ----------
    if getattr(args, "model", None) is not None:      # NEW: --model
        model.name = args.model

    # ---------- overrides: train ----------
    if getattr(args, "nodes", None) is not None:      train.nodes = args.nodes
    if getattr(args, "rounds", None) is not None:     train.rounds = args.rounds
    if getattr(args, "epochs", None) is not None:     train.epochs = args.epochs
    if getattr(args, "batch_size", None) is not None: train.batch_size = args.batch_size
    if getattr(args, "lr", None) is not None:         train.lr = args.lr
    if getattr(args, "target_acc", None) is not None: train.target_acc = args.target_acc
    if getattr(args, "upload_delta", False):          train.upload_delta = True
    if getattr(args, "save_updates", False):          train.save_updates = True

    # ---------- overrides: reward ----------
    if getattr(args, "base_reward", None) is not None:      reward.base_reward = args.base_reward
    if getattr(args, "stake_weight", None) is not None:     reward.stake_weight = args.stake_weight
    if getattr(args, "committee_size", None) is not None:   reward.committee_size = args.committee_size
    if getattr(args, "hist_decay", None) is not None:       reward.hist_decay = args.hist_decay
    if getattr(args, "reward_rate", None) is not None:      reward.reward_rate = args.reward_rate
    if getattr(args, "penalize_negative", False):           reward.penalize_negative = True
    # NEW:
    if getattr(args, "committee_bonus", None) is not None:  reward.committee_bonus = args.committee_bonus
    if getattr(args, "mal_eval_diff_thresh", None) is not None:
        reward.mal_eval_diff_thresh = args.mal_eval_diff_thresh

    # ---------- overrides: attack ----------
    if getattr(args, "malicious_frac", None) is not None:      attack.malicious_frac = args.malicious_frac
    if getattr(args, "malicious_strategy", None) is not None:  attack.malicious_strategy = args.malicious_strategy
    if getattr(args, "scale", None) is not None:               attack.scale = args.scale
    if getattr(args, "noise_std", None) is not None:           attack.noise_std = args.noise_std
    if getattr(args, "spoof_mode", None) is not None:          attack.spoof_mode = args.spoof_mode
    if getattr(args, "malicious_seed", None) is not None:      attack.seed = args.malicious_seed

    return Cfg(data=data, model=model, train=train, reward=reward, attack=attack, paths=paths)
