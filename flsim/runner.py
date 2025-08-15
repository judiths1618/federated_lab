
import csv, json
from pathlib import Path
from typing import List
import random

from .config import Cfg
from .partition import make_flower_partitions
from .node import LocalNode, NodeConfig
from .storage import IPFSSim
from .contracts import OurContract
from flsim.federated_aggregation.aggregator import Aggregator
from .evaluation import evaluate_global
from .attack import make_behavior

class FLRunner:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        for p in [cfg.paths.run_dir, cfg.paths.models_dir, cfg.paths.updates_dir, cfg.paths.sim_dir]:
            Path(p).mkdir(parents=True, exist_ok=True)
        self.ipfs = IPFSSim(); self.contract = OurContract()

    def _init_nodes(self, subsets, key_pairs) -> List[LocalNode]:
        nodes = []
        N = self.cfg.train.nodes
        mN = int(round(self.cfg.attack.malicious_frac * N))
        rng = random.Random(self.cfg.attack.seed)
        malicious_ids = set(rng.sample(list(range(N)), mN)) if mN > 0 else set()
        for i in range(N):
            behavior = make_behavior(
                strategy=(self.cfg.attack.malicious_strategy if i in malicious_ids else "none"),
                scale=self.cfg.attack.scale,
                noise_std=self.cfg.attack.noise_std,
                spoof_mode=self.cfg.attack.spoof_mode,
            )
            nodes.append(LocalNode(
                NodeConfig(i, self.cfg.train.batch_size, self.cfg.train.lr, self.cfg.train.epochs),
                subsets[i], key_pairs[i], self.ipfs, self.contract,
                upload_delta=self.cfg.train.upload_delta,
                save_updates=self.cfg.train.save_updates,
                save_dir=str(self.cfg.paths.run_dir),
                init_stake=100.0, init_reputation=10.0,
                behavior=behavior,
            ))
        return nodes

    def run(self):
        from .models import LinearMNIST

        # 0) 初始化全局模型 g0
        g0 = self.ipfs.save(LinearMNIST().state_dict())
        self.contract.set_global_model(0, g0)

        # 1) 数据划分 & 节点初始化
        subsets, key_pairs = make_flower_partitions(
            num_nodes=self.cfg.train.nodes,
            method=self.cfg.data.partitioner,
            alpha=self.cfg.data.alpha,
            classes_per_node=self.cfg.data.classes_per_node,
            shards_per_node=self.cfg.data.shards_per_node,
            samples_per_client=self.cfg.data.samples_per_client,
        )
        nodes = self._init_nodes(subsets, key_pairs)

        # [可选] 2) 合约参数注入 + 注册节点（如已 elsewhere 注入，可删除此块）
        try:
            self.contract.set_incentive_params(
                base_reward=self.cfg.reward.base_reward,
                stake_weight=self.cfg.reward.stake_weight,
                hist_decay=self.cfg.reward.hist_decay,
                rep_exponent=getattr(self.cfg.reward, "rep_exponent", 1.0),
                reward_rate=self.cfg.reward.reward_rate,
                penalize_negative=self.cfg.reward.penalize_negative,
                committee_bonus=getattr(self.cfg.reward, "committee_bonus", 0.10),
                mal_eval_diff_thresh=getattr(self.cfg.reward, "mal_eval_diff_thresh", 0.15),
            )
        except Exception:
            pass
        for n in nodes:
            try:
                self.contract.register_node(
                    n.cfg.node_id,
                    stake=getattr(n, "stake", 100.0),
                    reputation=getattr(n, "reputation", 10.0),
                )
            except Exception:
                pass

        # 3) 实例化 Aggregator（在循环外只建一次）
        aggr = Aggregator(
            self.ipfs, self.contract, nodes,
            save_dir=str(self.cfg.paths.run_dir),
            strategy_name=self.cfg.aggregation.strategy,

            # --- 评测提示（可选） ---
            dataset_name=getattr(getattr(self.cfg, "data", None), "name", "mnist"),
            model_name=getattr(getattr(self.cfg, "model", None), "name", None),

            # --- 激励/聚合相关（必需，取自 cfg.reward） ---
            reward_rate=self.cfg.reward.reward_rate,
            penalize_negative=self.cfg.reward.penalize_negative,
            base_reward=self.cfg.reward.base_reward,
            stake_weight=self.cfg.reward.stake_weight,
            committee_size=self.cfg.reward.committee_size,
            hist_decay_factor=self.cfg.reward.hist_decay,
        )


        # 4) 主循环
        with open(self.cfg.paths.log_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "round","global_acc","manifest_cid","node_id","samples","loss","acc",
                "claimed_acc","eval_acc","acc_diff",
                "update_type","model_update_cid","metrics_cid",
                "contrib_score","reward","balance","stake_penalty","stake","reputation",
                "malicious_detected","committee","is_malicious","strategy"
            ])

            for r in range(self.cfg.train.rounds):
                # 4.1) 取上一轮全局模型
                _, global_cid = self.contract.get_latest_global()

                # 4.2) 各节点本地训练，收集 manifest
                manifest = []
                for node in nodes:
                    loss, acc, upd_type, model_cid, metrics_cid = node.train_one_round(r, global_cid)
                    manifest.append({
                        "round": r, "node_id": node.cfg.node_id,
                        "samples": node.num_samples, "loss": loss, "acc": acc,
                        "update_type": upd_type, "model_update_cid": model_cid, "metrics_cid": metrics_cid,
                        "is_malicious": int(node.behavior.is_malicious), "strategy": node.behavior.strategy,
                    })

                # 4.3) 聚合（Aggregator 会把 features/committee/contrib 写给合约，并触发清算）
                new_cid, metrics_map, contrib_map, reward_map = aggr.aggregate_round(r, base_cid=global_cid)
                

                # 4.4) 保存 manifest（合约 + 本地）
                manifest_cid = self.ipfs.save(manifest)
                # 别名保持：你可以用 set_round_manifest 或 set_manifest
                self.contract.set_round_manifest(r, manifest_cid)
                with open(self.cfg.paths.sim_dir / f"round_{r}_manifest.json", "w") as mf:
                    json.dump(manifest, mf, indent=2)

                # 4.5) 评测新全局模型
                acc = evaluate_global(new_cid, self.ipfs)

                # 4.6) 写日志行
                for row in manifest:
                    nid = row["node_id"]

                    claimed_acc = float(metrics_map.get(nid, {}).get("acc", float("nan")))
                    eval_acc = float(self.contract.features.get(r, {}).get(nid, {}).get("eval_acc", float("nan")))
                    acc_diff = claimed_acc - eval_acc

                    contrib_score = float(self.contract.contribs.get(r, {}).get(nid, float("nan")))
                    reward = float(self.contract.rewards.get(r, {}).get(nid, 0.0))
                    balance = float(self.contract.get_balance(nid))
                    penalty = float(self.contract.penalties.get(r, {}).get(nid, 0.0))
                    stake = float(self.contract.nodes.get(nid, {}).get("stake", float("nan")))
                    reputation = float(self.contract.nodes.get(nid, {}).get("reputation", float("nan")))
                    mal_detected = self.contract.mal_detected.get(r, {}).get(nid, 0)
                    in_committee = int(nid in set(self.contract.committees.get(r, [])))
                    print(reward, balance, penalty, stake, reputation)
                    writer.writerow([
                        r,
                        f"{acc:.4f}",
                        manifest_cid,
                        nid,
                        row["samples"],
                        f"{row['loss']:.6f}",
                        f"{row['acc']:.6f}",
                        f"{claimed_acc:.4f}",
                        f"{eval_acc:.4f}",
                        f"{acc_diff:.4f}",
                        row["update_type"],
                        row["model_update_cid"],
                        row["metrics_cid"],
                        f"{contrib_score:.4f}",
                        f"{reward:.4f}",
                        f"{balance:.4f}",
                        f"{penalty:.4f}",
                        f"{stake:.4f}",
                        f"{reputation:.4f}",
                        mal_detected,
                        in_committee,
                        row["is_malicious"],
                        row["strategy"],
                    ])

                print(f"Round {r}: global acc={acc:.3f} | manifest={manifest_cid}")

                # 4.7) 每轮保存 acc_diff 直方图（用于阈值调参）
                try:
                    import numpy as _np, csv as _csv, os as _os
                    diffs = []
                    feats_r = self.contract.features.get(r, {})
                    for nid in feats_r:
                        eval_acc = feats_r.get(nid, {}).get("eval_acc", float("nan"))
                        claimed = metrics_map.get(nid, {}).get("acc", float("nan"))
                        if isinstance(eval_acc, float) and isinstance(claimed, float):
                            if not (eval_acc != eval_acc or claimed != claimed):  # 过滤 NaN
                                diffs.append(claimed - eval_acc)
                    bins = _np.linspace(-1.0, 1.0, 21)
                    counts, edges = _np.histogram(_np.asarray(diffs, dtype=float), bins=bins)
                    outp = _os.path.join(str(self.cfg.paths.run_dir), f"acc_diff_hist_round_{r}.csv")
                    with open(outp, "w", newline="") as hf:
                        w = _csv.writer(hf)
                        w.writerow(["bin_left","bin_right","count"])
                        for i in range(len(counts)):
                            w.writerow([f"{edges[i]:.4f}", f"{edges[i+1]:.4f}", int(counts[i])])
                except Exception:
                    pass

                # 4.8) 早停
                if self.cfg.train.target_acc > 0 and acc >= self.cfg.train.target_acc:
                    import torch
                    torch.save(self.ipfs.load(new_cid), self.cfg.paths.models_dir / f"final_global_round_{r}.pt")
                    print(f"[STOP] Target {self.cfg.train.target_acc} reached @ round {r}")
                    break

        print(f"Logs -> {self.cfg.paths.log_csv}")
        print(f"Models -> {self.cfg.paths.models_dir}")
        print(f"Manifests -> {self.cfg.paths.sim_dir}")
        print(f"Updates -> {self.cfg.paths.updates_dir}")
