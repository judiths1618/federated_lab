from typing import Tuple, List
import torchvision.transforms as T
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    IidPartitioner,
    DirichletPartitioner,
    PathologicalPartitioner,
    ShardPartitioner,
)

def build_partitioner(method: str, num_nodes: int, *, alpha: float, classes_per_node: int, shards_per_node: int, seed: int = 42):
    if method == "iid":
        return IidPartitioner(num_partitions=num_nodes)
    if method == "dirichlet":
        return DirichletPartitioner(
            num_partitions=num_nodes,
            partition_by="label",
            alpha=alpha,
            shuffle=True,
            seed=seed,
        )
    if method == "pathological":
        return PathologicalPartitioner(
            num_partitions=num_nodes,
            partition_by="label",
            num_classes_per_partition=classes_per_node,
            shuffle=True,
            seed=seed,
        )
    if method == "shards":
        return ShardPartitioner(
            num_partitions=num_nodes,
            partition_by="label",
            num_shards_per_partition=shards_per_node,
            shuffle=True,
            seed=seed,
        )
    raise ValueError(f"Unknown partitioner: {method}")

def hf_detect_keys(ds) -> Tuple[str, str]:
    if hasattr(ds, "features"):
        cols = set(ds.features.keys())
    else:
        cols = set(getattr(ds, "column_names", []))
    img_key = "image" if "image" in cols else ("img" if "img" in cols else next(iter(cols)))
    label_key = "label" if "label" in cols else ("labels" if "labels" in cols else None)
    if label_key is None:
        raise ValueError("Could not detect label key; expected 'label' or 'labels'.")
    return img_key, label_key

def to_torch_dataset(ds):
    img_key, label_key = hf_detect_keys(ds)
    to_tensor = T.ToTensor()
    def _apply(batch):
        batch[img_key] = [to_tensor(img) for img in batch[img_key]]
        return batch
    return ds.with_transform(_apply), img_key, label_key

def make_flower_partitions(
    num_nodes: int,
    method: str,
    *,
    alpha: float,
    classes_per_node: int,
    shards_per_node: int,
    samples_per_client: int = 0,
):
    # 分别构建两个独立的 partitioner 对象
    seed_train = 42
    seed_test = 42  # 如果想让 test 的划分随机性不同，可以改成 43 等
    train_part = build_partitioner(
        method,
        num_nodes,
        alpha=alpha,
        classes_per_node=classes_per_node,
        shards_per_node=shards_per_node,
        seed=seed_train,
    )
    test_part = build_partitioner(
        method,
        num_nodes,
        alpha=alpha,
        classes_per_node=classes_per_node,
        shards_per_node=shards_per_node,
        seed=seed_test,
    )

    fds = FederatedDataset(
        dataset="mnist",
        partitioners={"train": train_part, "test": test_part},
    )

    train_subsets: List = []
    test_subsets: List = []
    keys: List[Tuple[str, str]] = []
    for pid in range(num_nodes):
        train_ds = fds.load_partition(pid, "train")
        test_ds = fds.load_partition(pid, "test")
        if samples_per_client and len(train_ds) > samples_per_client:
            train_ds = train_ds.select(range(samples_per_client))
        train_torch, img_key, label_key = to_torch_dataset(train_ds)
        test_torch, _, _ = to_torch_dataset(test_ds)
        train_subsets.append(train_torch)
        test_subsets.append(test_torch)
        keys.append((img_key, label_key))
    return train_subsets, test_subsets, keys
