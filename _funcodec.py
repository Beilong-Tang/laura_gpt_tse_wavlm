from argparse import Namespace

import torch

from funcodec.train.distributed_utils import DistributedOption


def init_distributed_option(rank:int):
    distributed_option = DistributedOption()
    distributed_option.distributed = True
    distributed_option.dist_rank = rank
    distributed_option.local_rank = rank
    distributed_option.dist_world_size = torch.distributed.get_world_size()
    return distributed_option