import argparse
import os
import random
import torch.distributed
import yaml
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch

from _funcodec import init_distributed_option

from funcodec.tasks.text2audio_generation import Text2AudioGenTask
from funcodec.schedulers.warmup_lr import WarmupLR
from funcodec.torch_utils.load_pretrained_model import load_pretrained_model
from funcodec.train.distributed_utils import DistributedOption

from utils import setup_logger
from utils import init


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(4, 1)

    def forward(self, x):
        """x:[B,T]"""
        output = self.model(x)
        return output


def setup_seed(seed, rank):
    SEED = int(seed) + rank
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return SEED


## ddp process
def setup(rank, world_size, backend, port=12355):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, args):
    l = setup_logger(args, rank)
    l.info("logging initialized succesully")
    l.info(args)
    l.info(f"rank {rank} of world_size {len(args.gpus)} started...")
    setup(rank, len(args.gpus), args.dist_backend)
    args.gpu = args.gpus[rank]
    torch.cuda.set_device(args.gpu)
    torch.cuda.empty_cache()
    setup_seed(args.seed, rank)
    l.info("setup model")
    ## load laura gpt model
    model: nn.Module = Text2AudioGenTask.build_model(args)
    model.cuda()
    for p in args.init_param:
        l.info(f"Loading pretrained params from {p}")
        load_pretrained_model(
            model=model,
            init_param=p,
            ignore_init_mismatch=True,
            # NOTE(kamo): "cuda" for torch.load always indicates cuda:0
            #   in PyTorch<=1.4
            map_location=f"cuda:{torch.cuda.current_device()}",
        )
    model = DDP(model, device_ids=[args.gpu])
    l.info(f"model {model} is intialized")
    ## optimizer
    optim = init(torch.optim, args.optim, model.parameters())
    ## scheduler
    assert args.scheduler == "warmuplr"
    scheduler = WarmupLR(optim, **args.scheduler_conf)
    l.info(f"scheduler {scheduler} and optim {optim} is initialized")
    ## setup dataloader
    ### Initialize distributed Option to ensure compability
    distributed_option = init_distributed_option(rank)



    

    ct = 0
    while True:
        data = torch.randn(1, 4)
        data = data.to(rank)
        loss = model(data)
        loss.backward()
        if ct % 10 == 0:
            print(f"loss on rank {rank} is {loss}")
            l.info(f"loss on rank {rank} is {loss}")
        ct += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument("--gpus", default="0,1,2,3", type=str, help="gpus")
    parser.add_argument("--log", default="./log", type=str, help="Output of the log")
    parser.add_argument("--config", type=str, default=None, help="path to yaml config")
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
        for k, v in config.items():
            args.__setattr__(k, v)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.gpus = [int(i) for i in args.gpus.split(",")]
    mp.spawn(main, args=(args,), nprocs=len(args.gpus), join=True)
    pass
