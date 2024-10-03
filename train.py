import argparse
import os

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch
from utils import setup_logger

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(4, 1)

    def forward(self, x):
        """x:[B,T]"""
        output = self.model(x)
        return output


## ddp process
def setup(rank, world_size, backend, port=12355):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, args):
    logger = setup_logger(args)
    logger.info("logging initialized succesully")
    print(f"rank {rank} of world_size {len(args.gpus)} started...")
    setup(rank, len(args.gpus), args.dist_backend)
    args.gpu = args.gpus[rank]
    torch.cuda.set_device(args.gpu)
    model = Model()
    model.cuda()
    model = DDP(model, device_ids=[args.gpu])

    ct = 0
    while True:
        data = torch.randn(1, 4)
        data = data.to(rank)
        loss = model(data)
        loss.backward()
        if ct % 10 == 0:
            print(f"loss on rank {rank} is {loss}")
            logger.info(f"loss on rank {rank} is {loss}")
        ct += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument("--gpus", default="0,1,2,3", type="str", help="gpus")
    parser.add_argument("--log", default="./log", type=str, help="Output of the log")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.gpus = [int(i) for i in args.gpus.split(",")]
    mp.spawn(main, args=(args,), nprocs=len(args.gpus), join=True)
    pass
