import torch
import einops

def sample_vectors(samples, num):
    # samples.shape = cik
    # num = j
    assert samples.ndim == 3
    device = samples.device
    # num_samples = i
    num_codebooks = samples.shape[0]
    num_samples = samples.shape[1]
    # indices.shape = cj
    indices = torch.randint(
        0,
        num_samples,
        (
            num_codebooks,
            num,
        ),
        device=device,
    )
    # samples[indices].shape = cjk
    indices = einops.repeat(indices, "c j -> c j k", k=samples.shape[2])
    return torch.gather(samples, dim=1, index=indices)

def batched_bincount(x, dim, max_value):
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target

def kmeans(samples, num_clusters, num_iters=10):
    # samples.shape = cik
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device
    num_codebooks = samples.shape[0]
    # means.shape = cjk
    means = sample_vectors(samples, num_clusters)
    bins = None
    for _ in range(num_iters):
        # diffs.shape = cijk
        diffs = einops.rearrange(samples, "c i k -> c i () k") - einops.rearrange(
            means, "c j k -> c () j k"
        )
        # diffs.shape = cij
        dists = -(diffs ** 2).sum(dim=-1)
        # buckets.shape = ci
        buckets = dists.max(dim=-1).indices
        # bins.shape = cj
        bins = batched_bincount(buckets, dim=-1, max_value=num_clusters)
        # zero_mask.shape = cj
        zero_mask = bins == 0
        # bins_min_clamped.shape = cj
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        # new_means.shape = cjk
        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(1, einops.repeat(buckets, "c i -> c i k", k=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        # means.shape = cjk
        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


def get_keys_by_labels_kmeans(inputs, labels, num_clusters):
    out = []
    for value in torch.unique(labels):
        indices = torch.nonzero(labels == value).squeeze()
        tensor = inputs[indices,:, :]
        out.append(tensor)
        centroids = kmeans(tensor, num_clusters=num_clusters)
        
    return out
#kmeans #nextstep: replace kmeans with OT
#get keys: