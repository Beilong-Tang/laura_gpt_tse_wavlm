import torch
import math
import random


def cosine_schedule(r: float):
    """
    :return cos(pi/2*r)
    """
    assert r >= 0 and r <= 1
    return math.cos(math.pi / 2 * r)


def random_ratio():
    return random.uniform(0, 1)


def sample_cosine():
    return cosine_schedule(random.uniform(0, 1))


def replace_mask(codegram: torch.Tensor, ratio: float, mask_value=1024):
    """
    :params codegram: [B, N(9), T]
    :params ratio: the ratio to mask, between [0,1]
    :return the replaced codegram
    :return the index of shape [B, num]
    """
    codegram_ = codegram.clone()
    assert len(codegram_.shape) == 3
    assert ratio <= 1 and ratio >= 0
    b, n, t = codegram_.shape
    total = n * t
    num_items_to_replace_per_batch = int(ratio * total)
    if num_items_to_replace_per_batch == 0:
        ## add this to avoid nan problem in the back propagation
        num_items_to_replace_per_batch = 1

    ## first approach
    flat_indices = (
        torch.randperm(codegram[0].numel())[:num_items_to_replace_per_batch]
        .unsqueeze(0)
        .repeat(b, 1)
    )  # [b, n]
    rows = flat_indices // t
    cols = flat_indices % t
    batch_indices = torch.arange(b).unsqueeze(1).expand_as(rows)
    codegram_[batch_indices, rows, cols] = mask_value

    ## workaround #2 -> O(nlogn)
    # flat_indices = torch.randn(b, total).argsort(dim=1)[
    #     :, :num_items_to_replace_per_batch
    # ]  # [b, n]
    # rows = flat_indices // t  # [b, n]
    # cols = flat_indices % t  # [b, n]
    # batch_indices = torch.arange(b).unsqueeze(1).expand_as(rows)
    # codegram_[batch_indices, rows, cols] = mask_value

    return codegram_, flat_indices


def take_index(codegram, flat_indices):
    """
    :params: codegram: [B, N,T] or [B, C, N, T]
    :params: flat_indices: the indices from replace_mask() [b, num]
    :return: [B, num] or [B, C, num]
    """
    t = codegram.size(-1)
    b = codegram.size(0)
    rows = flat_indices // t  # [b, n]
    cols = flat_indices % t  # [b, n]
    batch_indices = torch.arange(b).unsqueeze(1).expand_as(rows)
    if len(codegram.shape) == 4:
        # [B, C, N ,T]
        return codegram[batch_indices, :, rows, cols].permute(0, 2, 1)
    else:
        return codegram[batch_indices, rows, cols]


def get_unmask_index(codegram, mask_value=1024):
    """
    :params: codegram: [1, n, T]
    :params: value: int, the value to get the index
    :return: index: the index indicating the unmasked area
    """
    codegram = codegram.squeeze(0)  # [n, T]
    index = (codegram != mask_value).nonzero()
    return index


def set_value(tensor, index, value):
    """
    :params: tensor: [N, T]
    :params: index: index from get_unmask_index()
    :params: value: the value for the tensor to be replaced to
    :return: tensor: [N, T]
    """
    tensor[index[:, 0], index[:, 1]] = value
    return tensor
