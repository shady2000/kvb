import torch
def partition_input_by_labels(inputs, labels):
    out = []
    labels = torch.tensor([0, 1, 2, 3, 5, 5, 2, 3, 4, 1, 2, 3, 4, 2, 3])
    for value in torch.unique(labels):
        indices = torch.nonzero(labels == value).squeeze()
        tensor = inputs[indices,:, :]
        out.append(tensor)
    return out

keys, cluster_size = kmeans(
    data, self.key_value_pairs_per_codebook, self.kmeans_iters
)