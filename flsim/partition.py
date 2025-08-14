from typing import Tuple, List
import torchvision.transforms as T
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    IidPartitioner, DirichletPartitioner, PathologicalPartitioner, ShardPartitioner,
)

def build_partitioner(method: str, num_nodes: int, *, alpha: float, classes_per_node: int, shards_per_node: int):
    if method == "iid":
        # return IidPartitioner(num_partitions=num_nodes, shuffle=True, seed=42)
        return IidPartitioner(num_partitions=num_nodes)
    if method == "dirichlet":
        return DirichletPartitioner(num_partitions=num_nodes, partition_by="label", alpha=alpha, shuffle=True, seed=42)
    if method == "pathological":
        return PathologicalPartitioner(num_partitions=num_nodes, partition_by="label", num_classes_per_partition=classes_per_node, shuffle=True, seed=42)
    if method == "shards":
        return ShardPartitioner(num_partitions=num_nodes, partition_by="label", num_shards_per_partition=shards_per_node, shuffle=True, seed=42)
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

def make_flower_partitions(num_nodes: int, method: str, *, alpha: float, classes_per_node: int, shards_per_node: int, samples_per_client: int = 0):
    fds = FederatedDataset(dataset="mnist", partitioners={"train": build_partitioner(method, num_nodes, alpha=alpha, classes_per_node=classes_per_node, shards_per_node=shards_per_node)})
    subsets = []
    keys = []
    for pid in range(num_nodes):
        ds = fds.load_partition(pid, "train")
        if samples_per_client and len(ds) > samples_per_client:
            ds = ds.select(range(samples_per_client))
        ds_torch, img_key, label_key = to_torch_dataset(ds)
        subsets.append(ds_torch)
        keys.append((img_key, label_key))
    return subsets, keys
