#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

from flsim.config import build_config
from flsim.runner import FLRunner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=str, default="baseline", help="Experiment name (folder under runs)")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--nodes", type=int, default=None)
    ap.add_argument("--target-acc", type=float, default=None)
    ap.add_argument("--partitioner", type=str, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--classes-per-node", type=int, default=None)
    ap.add_argument("--shards-per-node", type=int, default=None)
    ap.add_argument("--samples-per-client", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--upload-delta", action="store_true")
    ap.add_argument("--save-updates", action="store_true")
    # reward/penalty
    ap.add_argument("--base-reward", type=float, default=None)
    ap.add_argument("--stake-weight", type=float, default=None)
    ap.add_argument("--committee-size", type=int, default=None)
    ap.add_argument("--hist-decay", type=float, default=None)
    ap.add_argument("--reward-rate", type=float, default=None)
    ap.add_argument("--penalize-negative", action="store_true")
    ap.add_argument("--agg-strategy", type=str, default=None, help="Aggregation strategy (e.g., fedavg)")
    # attacks
    ap.add_argument("--malicious-frac", type=float, default=None, help="0.0~1.0 fraction of malicious clients")
    ap.add_argument("--malicious-strategy", type=str, default=None, choices=["none","signflip","scaling","gaussian","label_flip","metric_spoof"])
    ap.add_argument("--scale", type=float, default=None, help="scale magnitude for signflip/scaling")
    ap.add_argument("--noise-std", type=float, default=None, help="std for gaussian noise on updates")
    ap.add_argument("--spoof-mode", type=str, default=None, choices=["none","high_acc","low_acc","random"], help="how metrics are spoofed by malicious")
    ap.add_argument("--malicious-seed", type=int, default=None)

    args = ap.parse_args()

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / args.exp / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config(args, run_dir)
    runner = FLRunner(cfg)
    runner.run()

if __name__ == "__main__":
    main()
