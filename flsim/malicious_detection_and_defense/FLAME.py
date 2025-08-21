"""FLAME defense implementation.

This module implements a simplified version of the FLAME algorithm for
malicious update detection and model aggregation in federated learning.

The algorithm performs the following steps:

1. **Clustering** of client updates using HDBSCAN with cosine similarity.
2. **Median norm clipping** of client updates.
3. **Aggregation** of updates that belong to the main cluster.
4. **Noise addition** to the aggregated global model to improve privacy.

The implementation follows the pseudocode provided in the task
description.  It is intended as a reference implementation rather than a
fully‑featured production component.
"""

from __future__ import annotations

from typing import Dict, List

import torch

import numpy as np

import copy

try:  # pragma: no cover - optional dependency
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover
    hdbscan = None
    
def no_defence_balance(params, global_parameters):
    total_num = len(params)
    sum_parameters = None
    for i in range(total_num):
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in params[i].items():
                sum_parameters[key] = var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + params[i][var]
    for var in global_parameters:
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
            continue
        global_parameters[var] += (sum_parameters[var] / total_num)

    return global_parameters

def parameters_dict_to_vector(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] != 'weight' and key.split('.')[-1] != 'bias':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def flame(local_model, update_params, global_model, args, debug=False):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list=[]
    local_model_vector = []
    for param in local_model:
        # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
        local_model_vector.append(parameters_dict_to_vector_flt(param))
    for i in range(len(local_model_vector)):
        cos_i = []
        for j in range(len(local_model_vector)):
            cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)
    if debug==True:
        filename = './' + args.save + '/flame_analysis.txt'
        f = open(filename, "a")
        for i in cos_list:
            f.write(str(i))
            print(i)
            f.write('\n')
        f.write('\n')
        f.write("--------Round--------")
        f.write('\n')
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
    print(clusterer.labels_)
    benign_client = []
    norm_list = np.array([])

    max_num_in_cluster=0
    max_cluster_index=0
    if clusterer.labels_.max() < 0:
        for i in range(len(local_model)):
            benign_client.append(i)
            norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
                norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())  # no consider BN
    print(benign_client)
   
    for i in range(len(benign_client)):
        if benign_client[i] < num_malicious_clients:
            args.wrong_mal+=1
        else:
            #  minus per benign in cluster
            args.right_ben += 1
    args.turn+=1

    clip_value = np.median(norm_list)
    for i in range(len(benign_client)):
        gama = clip_value/norm_list[i]
        if gama < 1:
            for key in update_params[benign_client[i]]:
                if key.split('.')[-1] == 'num_batches_tracked':
                    continue
                update_params[benign_client[i]][key] *= gama
    
    global_model = no_defence_balance([update_params[i] for i in benign_client], global_model)
    #add noise
    for key, var in global_model.items():
        if key.split('.')[-1] == 'num_batches_tracked':
                    continue
        temp = copy.deepcopy(var)
        temp = temp.normal_(mean=0,std=args.noise*clip_value)
        var += temp
    return global_model


def flame_analysis(local_model, args, debug=False):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list=[]
    local_model_vector = []
    for param in local_model:
        local_model_vector.append(parameters_dict_to_vector_flt(param))
    for i in range(len(local_model_vector)):
        cos_i = []
        for j in range(len(local_model_vector)):
            cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)
    if debug==True:
        filename = './' + args.save + '/flame_analysis.txt'
        f = open(filename, "a")
        for i in cos_list:
            f.write(str(i))
            f.write('/n')
        f.write('/n')
        f.write("--------Round--------")
        f.write('/n')
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
    print(clusterer.labels_)
    benign_client = []

    max_num_in_cluster=0
    max_cluster_index=0
    if clusterer.labels_.max() < 0:
        for i in range(len(local_model)):
            benign_client.append(i)
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
    return benign_client

class FlameDefense:
    """Apply the FLAME defense to a set of client model weights.

    Parameters
    ----------
    global_model: ``torch.nn.Module``
        The model that will be updated in‑place with the aggregated client
        updates.
    defense: bool, optional
        If ``False`` the class will still aggregate the client updates but
        will skip all FLAME‑specific defences.
    """

    def __init__(self, global_model: torch.nn.Module, defense: bool = True) -> None:
        self.global_model = global_model
        self.conf = {"defense": "flame" if defense else None}

    @staticmethod
    def _flatten_state(state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([p.reshape(-1) for p in state.values()])

    def defend(self, clients_weight: List[Dict[str, torch.Tensor]]) -> None:
        """Run the FLAME algorithm and update ``self.global_model`` in‑place."""
        if not clients_weight:
            return

        if hdbscan is None:
            raise ImportError("hdbscan is required for FlameDefense")

        # ----- 1. clustering -----
        weight_vectors = [self._flatten_state(s) for s in clients_weight]
        clients_weight_total = torch.stack(weight_vectors).double()
        num_clients = clients_weight_total.shape[0]

        cluster = hdbscan.HDBSCAN(
            metric="cosine",
            algorithm="generic",
            min_cluster_size=num_clients // 2 + 1,
            min_samples=1,
            allow_single_cluster=True,
        )
        cluster.fit(clients_weight_total)

        # ----- 2. median norm clipping -----
        euclidean = torch.norm(clients_weight_total, p=2, dim=1)
        med = torch.median(euclidean)
        for i, data in enumerate(clients_weight):
            gamma = med / euclidean[i]
            gamma = torch.clamp(gamma, max=1.0)
            for name, params in data.items():
                params.data = (params.data * gamma).to(params.data.dtype)

        # ----- 3. aggregation -----
        weight_accumulator: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param, dtype=torch.double)
            for name, param in self.global_model.state_dict().items()
        }

        num_in = 0
        for i, data in enumerate(clients_weight):
            if self.conf["defense"] == "flame" and cluster.labels_[i] == 0:
                num_in += 1
                for name, params in data.items():
                    weight_accumulator[name].add_(params.double())

        self.model_aggregate(weight_accumulator, max(num_in, 1))

        # ----- 4. noise addition -----
        if self.conf["defense"] == "flame":
            lamda = 0.000012
            for name, param in self.global_model.named_parameters():
                if "bias" in name or "bn" in name:
                    continue
                std = lamda * med * param.data.std()
                noise = torch.normal(0, std, size=param.size(), device=param.device)
                param.data.add_(noise)

    def model_aggregate(self, weight_accumulator: Dict[str, torch.Tensor], num: int) -> None:
        """Aggregate ``weight_accumulator`` into ``self.global_model``."""
        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name] / float(num)
            if data.dtype != update_per_layer.dtype:
                data.add_(update_per_layer.to(data.dtype))
            else:
                data.add_(update_per_layer)
