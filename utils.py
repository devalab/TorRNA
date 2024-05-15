import torch

def get_indices_of_last_node_from_batch(batch):
    node_batch_idx = batch.batch

    last_node_indices = list()

    cur_batch = 0
    for idx, each_node_batch in enumerate(node_batch_idx):
        if each_node_batch.item() != cur_batch:
            last_node_indices.append(idx-1)
            cur_batch = each_node_batch.item()
    last_node_indices.append(len(node_batch_idx)-1)

    return torch.tensor(last_node_indices)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))