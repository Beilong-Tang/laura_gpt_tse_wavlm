import argparse
import torch.multiprocessing as mp
import torch
import os.path as op
import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import numpy as np

from utils.seed import setup_seed
from utils.logger import setup_logger
from utils.load_scp import get_source_list

from models.wavlm.WavLMWrapper import WavLMWrapper as WavLM
import torchaudio
import tqdm
import time


SEED = 1234

LOG_INTERVAL = 100


def extract_data(rank, args):
    device = args.gpus[rank % len(args.gpus)]
    logger = setup_logger(args, rank=rank, out=False)
    if args.only_clean:
        logger.info("only infering on clean dataset")
        type_dict = {"clean": args.clean_scp}
    else:
        type_dict = {"clean": args.clean_scp, "mix": args.mix_scp}
    wavlm = WavLM(args.wavlm_ckpt).to(device)
    for k, v in type_dict.items():
        base_dir = op.join(args.output_dir, k, str(rank))
        os.makedirs(base_dir, exist_ok=True)
        scp_writter = open(op.join(args.output_dir, k, f"{str(rank)}.scp"), "w")
        logger.info(f"output to {base_dir}")
        names, paths = get_source_list(v, ret_name=True)
        names = names[rank :: args.num_proc]
        paths = paths[rank :: args.num_proc]
        ct = 0
        start_time = time.time()
        for n, p in tqdm.tqdm(list(zip(names, paths))):
            audio, _ = torchaudio.load(p)  # [1,T]
            audio = audio.to(device)
            ## Extract the 6th layer output
            emb: torch.Tensor = wavlm(audio)  # [1,T,E]
            emb = emb.squeeze(0)  # [T, E]
            tt = emb.size(0)
            emb = emb.cpu().numpy()
            save_path = op.join(base_dir, f"{n}.npz")
            np.save(save_path, emb)
            scp_writter.write(f"{n} {save_path} {str(tt)}\n")
            if ct % LOG_INTERVAL == 0:
                time_left = ((time.time() - start_time) / LOG_INTERVAL) * (
                    len(names) - ct
                )
                info = f"rank {rank} finishes {ct}/{len(names)}, estimated time left: {time_left}"
                logger.info(info)
                print(info)
                start_time = time.time()
            ct += 1
            pass
        pass
    pass


def main(args):
    # os.makedirs(args.)
    setup_seed(SEED)
    mp.spawn(extract_data, args=(args,), nprocs=args.num_proc, join=True)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_scp", type=str, required=True)
    parser.add_argument("--mix_scp", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--wavlm_ckpt", type=str, required=True)
    parser.add_argument("--only_clean", action="store_true")
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument(
        "--gpus", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    )
    args = parser.parse_args()
    main(args)
    pass
