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


# def _extract(type_dict: dict, rank: int, logger, device: str, wavlm, args):
#     for k, v in type_dict.items():
#         base_dir = op.join(args.output_dir, k, str(rank))
#         os.makedirs(base_dir, exist_ok=True)
#         scp_writter = open(op.join(args.output_dir, k, f"{str(rank)}.scp"), "w")
#         logger.info(f"output to {base_dir}")
#         names, paths = get_source_list(v, ret_name=True)
#         names = names[rank :: args.num_proc]
#         paths = paths[rank :: args.num_proc]
#         ct = 0
#         start_time = time.time()
#         for n, p in tqdm.tqdm(list(zip(names, paths))):
#             audio, _ = torchaudio.load(p)  # [1,T]
#             audio = audio.to(device)
#             ## Extract the 6th layer output
#             emb: torch.Tensor = wavlm(audio)  # [1,T,E]
#             emb = emb.squeeze(0)  # [T, E]
#             tt = emb.size(0)
#             emb = emb.cpu().numpy()
#             save_path = op.join(base_dir, f"{n}.npz")
#             np.save(save_path, emb)
#             scp_writter.write(f"{n} {save_path} {str(tt)}/n")
#             if ct % LOG_INTERVAL == 0:
#                 time_left = ((time.time() - start_time) / LOG_INTERVAL) * (
#                     len(names) - ct
#                 )
#                 info = f"|{k}| rank {rank} finishes {ct}/{len(names)}, estimated time left: {time_left}"
#                 logger.info(info)
#                 print(info)
#                 start_time = time.time()
#             ct += 1
#             pass
#         pass


def _extract_target(path_dict: dict, rank: int, logger, device: str, wavlm, args):
    """
    path_dict: {name: [mix, ref]}
    """
    ref_base_dir = op.join(args.output_dir, "aux", str(rank))
    mix_base_dir = op.join(args.output_dir, "mix", str(rank))
    os.makedirs(ref_base_dir, exist_ok=True)
    os.makedirs(mix_base_dir, exist_ok=True)
    scp_ref_writter = open(op.join(args.output_dir, "aux", f"{str(rank)}.scp"), "w")
    scp_mix_writter = open(op.join(args.output_dir, "mix", f"{str(rank)}.scp"), "w")
    logger.info(f"ref output to {ref_base_dir}")
    logger.info(f"mix output to {mix_base_dir}")
    items = list(path_dict.items())[rank :: args.num_proc]
    for name, (m_path, r_path) in tqdm.tqdm(items):
        r_audio = torchaudio.load(r_path)[0].to(device)  # [1,T]
        emb_r = wavlm(r_audio).squeeze(0).cpu().numpy()  # np [T,E]
        tt_r = emb_r.shape[0]  # T
        np.save(op.join(ref_base_dir, f"{name}.npy"), emb_r)
        scp_ref_writter.write(
            f"{name} {op.join(ref_base_dir, f'{name}.npy')} {str(tt_r)}\n"
        )
        m_audio = torchaudio.load(m_path)[0].to(device)  # [1,T]
        cat_audio = torch.cat([r_audio, m_audio, r_audio], dim=1)  # [1, T_r + T + T_r]
        emb_cat: torch.Tensor = wavlm(cat_audio).squeeze(0)  # [T', E]
        total_t = emb_cat.size(0)
        emb_mix = emb_cat[tt_r : total_t - tt_r].cpu().numpy()  # np [T',E]
        np.save(op.join(mix_base_dir, f"{name}.npy"), emb_mix)
        scp_mix_writter.write(
            f"{name} {op.join(mix_base_dir, f'{name}.npy')} {str(emb_mix.shape[0])}\n"
        )


def extract_data(rank, args):
    device = args.gpus[rank % len(args.gpus)]
    logger = setup_logger(args, rank=rank, out=False)
    wavlm = WavLM(args.wavlm_ckpt).to(device)
    # Extract mixed wavlm, contanetaed with auxilary register audio
    mix_scp, aux_scp = args.mix_scp, args.aux_scp
    mix_names, mix_paths = get_source_list(mix_scp, ret_name=True)
    aux_names, aux_paths = get_source_list(aux_scp, ret_name=True)
    path_dict = {}
    for i in range(0, len(mix_names)):
        name = mix_names[i]
        m_p = mix_paths[i]
        r_p = aux_paths[aux_names.index(name)]
        path_dict[name] = [m_p, r_p]
    _extract_target(path_dict, rank, logger, device, wavlm, args)
    logger.info(f"Done!")


def main(args):
    # os.makedirs(args.)
    setup_seed(SEED)
    mp.spawn(extract_data, args=(args,), nprocs=args.num_proc, join=True)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aux_scp", type=str, required=True)
    parser.add_argument("--mix_scp", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--wavlm_ckpt", type=str, required=True)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument(
        "--gpus", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    )
    args = parser.parse_args()
    main(args)
    pass
