## Inference python scripts
import os
import sys
import argparse
import logging
import torch
import tqdm

sys.path.append(os.getcwd())

from funcodec.bin.text2audio_inference import save_audio
from utils import (
    setup_logger,
    update_args,
    setup_seed,
    init,
    strip_ddp_state_dict,
    AttrDict,
)

from decoder.wavlm_kmeans_conformer import WavLMKmeansConformer

from blpytorch.models.wavlm.WavLMWrapper import WavLMWrapper as WavLM
from blpytorch.data.target_dataset import TargetDataset


# Function to partition an IterableDataset for distributed processing
def partition_iterable_dataloader(dataloader, rank, world_size):
    for idx, batch in enumerate(dataloader):
        # Only process the batches corresponding to this rank
        if idx % world_size == rank:
            yield batch


@torch.no_grad()
def inference(args: argparse.Namespace):
    l: logging.Logger = args.logging
    args = AttrDict(**vars(args))
    os.makedirs(args.output_dir, exist_ok=True)
    ## load model
    ckpt = torch.load(args.model_file)
    model: torch.nn.Module = init(args.model)
    model.load_state_dict(ckpt["model_state_dict"])
    model.cuda()
    model.eval()
    l.info("model successfully intialized")
    wavlm = WavLM(ckpt_path=args.wavlm_path)
    wavlm.cuda()
    ## load decoder:
    if args.decoder is not None:
        d_conf = args.decoder
        decoder = WavLMKmeansConformer(
            kmeans_path=d_conf["kmeans_ckpt"],
            kernel_size=d_conf["kernel_size"],
            hifi_path=d_conf["hifi_path"],
            hifi_config=d_conf["hifi_config"],
        )
        d_ckpt = strip_ddp_state_dict(
            torch.load(d_conf["conformer_ckpt"])["model_state_dict"]
        )
        decoder.load_state_dict(d_ckpt, strict=False)
        decoder.cuda()
        decoder.eval()
    else:
        l.info("using ref decoder")
        lm = init(args.lm_model)
        film = init(args.FiLM)
        fusion = init(args.cross_attention_model)
        decoder = init(args.ref_decoder, lm_model=lm, fusion=fusion, film=film)
        d_ckpt = strip_ddp_state_dict(
            torch.load(args.decoder_ckpt, map_location="cpu")["model_state_dict"]
        )
        decoder.load_state_dict(d_ckpt, strict=False)
        decoder.cuda()
        decoder.eval()
        pass

    test_data = TargetDataset(
        mix_path=args.data["mix_path"],
        regi_path=args.data["regi_path"],
        clean_path=args.data["clean_path"],
        rank=-1,
        mix_length=None,
        regi_length=None,
        _type="audio",
    )

    l.info("data initialized successfully")
    for mix, _, regi, mix_path, _, _ in tqdm.tqdm(test_data):
        torch.cuda.empty_cache()
        mix, regi = mix.cuda(), regi.cuda()  # [T]
        mix, regi = mix.unsqueeze(0), regi.unsqueeze(0)  # [1,T]
        regi_mix = wavlm(torch.cat([regi, mix, regi], dim=1))  # [1, T' + T + T', E]
        regi_mix = regi_mix[:, : regi_mix.size(1) - regi.size(1)].unsqueeze(
            0
        )  # [T' + T, E]
        regi = wavlm(regi)  # [1, T, E]
        base_name = os.path.basename(mix_path)
        logging.info(f"generating {base_name}")
        continual = model.kmeans(regi).squeeze(0).unsqueeze(-1).tolist()  # list [T, 1]
        ret_val = model.decode_codec(
            text=regi_mix, aux=None, continual=continual
        )  # [1, T, 1]
        if ret_val.size(1) == 0:
            print(f"not generating audio for ret_val {base_name}")
            continue
        ret_val = ret_val.squeeze(-1)  # [1,T]
        l.info(f"ret_val: {ret_val.shape}")
        if args.decoder is not None:
            audio = decoder.inference(ret_val)  # [1,T']
        else:
            audio = decoder.recon_audio(model.kmeans.emb(ret_val), regi)
        save_path = os.path.join(args.output_dir, base_name)
        save_audio(audio, save_path, 16000, rescale=True)
    l.info("inferencing is done!")


def main(args: argparse.Namespace):
    setup_seed(args.seed, 0)
    logger = setup_logger(args, rank=0, out=False)
    args.logging = logger
    logger.info(args)
    inference(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--wavlm_path", type=str)
    parser.add_argument("--default_config", type=str)
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--raw_inputs", nargs="*", default=None, type=str)
    parser.add_argument("--device", default="cuda:4", type=str)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    update_args(args, args.config_file)
    update_args(args, args.default_config)
    main(args)
    pass
