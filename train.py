import argparse
import os
import random
import numpy as np
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from _funcodec import init_sequence_iter_factory

from trainer.abs_trainer import Trainer
from utils import setup_logger
from utils import init
from utils import AttrDict
from utils import update_args


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
    args = AttrDict(**vars(args))
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
    model: nn.Module = init(args.model)
    model.cuda()
    model = DDP(model, device_ids=[args.gpu])
    l.info(f"model {model} is intialized")
    ## optimizer
    optim = init(args.optim, model.parameters())
    ## scheduler
    
    scheduler = init(args.scheduler, optim)
    l.info(f"scheduler {scheduler} and optim {optim} is initialized")
    ## setup dataloader
    ### Initialized iter factory
    train_iter = init_sequence_iter_factory(args, rank, "train")
    val_iter = init_sequence_iter_factory(args, rank, "valid")

    ## ckpt_dir
    config_name = os.path.basename(args.config).replace(".yaml", "")
    ckpt_dir = Path(args.config).absolute().parent.parent.joinpath("ckpt").joinpath(config_name)
    os.makedirs(ckpt_dir, exist_ok= True)

    trainer = Trainer(
        model,
        train_iter,
        val_iter,
        optim,
        scheduler,
        config=args,
        ckpt_dir=ckpt_dir,
        rank=rank,
        logger=l,
    )
    l.info("starting training!")
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument("--gpus", default="0,1,2,3", type=str, help="gpus")
    parser.add_argument("--log", default="./log", type=str, help="Output of the log")
    parser.add_argument("--config", type=str, default=None, help="path to yaml config")
    args = parser.parse_args()
    update_args(args,args.config)
    log = setup_logger(args, rank = 0)
    log.info(f"torch cuda available: {torch.cuda.is_available()}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.gpus = [int(i) for i in args.gpus.split(",")]
    args.ngpu = len(args.gpus)
    mp.spawn(main, args=(args,), nprocs=len(args.gpus), join=True)
    pass
