## Inference python scripts
import os
import sys
import argparse
import logging
import torch
import librosa
import numpy as np
import tqdm

sys.path.append(os.getcwd())

from typing import Sequence
from typing import Union
from typing import Tuple
from typing import Optional

from funcodec.bin.text2audio_inference import Text2Audio, save_audio
from funcodec.tasks.text2audio_generation import Text2AudioGenTask
from utils import setup_logger, update_args, setup_seed, init, strip_ddp_state_dict, AttrDict

from decoder.wavlm_kmeans_conformer import WavLMKmeansConformer
from decoder.ref_conformer import ReferenceCrossAttention




def inference_func(
    output_dir: Optional[str] = None,
    batch_size: int = 1,
    dtype: str = "float32",
    device: str = "cuda",
    logging: logging.Logger = logging.getLogger(),
    num_workers: int = 0,
    key_file: Optional[str] = None,
    config_file: Optional[str] = "config.yaml",
    model_file: Optional[str] = "model.pth",
    model_tag: Optional[str] = None,
    allow_variable_data_keys: bool = True,
    streaming: bool = False,
    **kwargs,
):
    """
    copied from funcodec.bin.text2audio_inference.inference_func
    """

    # 2. Build model
    model_kwargs = dict(
        config_file=config_file,
        model_file=model_file,
        device=device,
        dtype=dtype,
        streaming=streaming,
        **kwargs,
    )
    my_model = Text2Audio.from_pretrained(
        model_tag=model_tag,
        **model_kwargs,
    )
    my_model.model.eval()

    def _forward(
        data_path_and_name_and_type: Sequence[Tuple[str, str, str]] = None,
        raw_inputs: Union[Tuple[str], Tuple[str, str, str]] = None,
        output_dir_v2: Optional[str] = None,
        param_dict: Optional[dict] = None,
    ):
        logging.info("param_dict: {}".format(param_dict))
        if data_path_and_name_and_type is None and raw_inputs is not None:
            # add additional parenthesis to keep the same data format as streaming_iterator
            logging.info("infering on one audio data")
            data_dict = dict(text=[raw_inputs[0]])
            if len(raw_inputs) == 3:
                data_dict["prompt_text"] = [raw_inputs[1]]
                if isinstance(raw_inputs[2], str):
                    data_dict["prompt_audio"] = [
                        librosa.load(
                            raw_inputs[2],
                            sr=my_model.codec_model.model.quantizer.sampling_rate,
                            mono=True,
                            dtype=np.float32,
                        )[0][np.newaxis, :]
                    ]
                else:
                    data_dict["prompt_audio"] = [raw_inputs[2].squeeze()[None, :]]
            loader = [(["utt1"], data_dict)]
        else:
            loader = Text2AudioGenTask.build_streaming_iterator(
                data_path_and_name_and_type,
                dtype=dtype,
                batch_size=batch_size,
                key_file=key_file,
                num_workers=num_workers,
                preprocess_fn=None,
                collate_fn=Text2AudioGenTask.build_collate_fn(
                    my_model.model_args, False, raw_sequence=("text", "prompt_text")
                ),
                allow_variable_data_keys=allow_variable_data_keys,
                inference=True,
            )

        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        result_list = []

        for keys, data in loader:
            key = keys[0]
            logging.info(f"generating {key}")
            model_inputs = [data["text"][0]]
            for input_key in ["prompt_text", "prompt_audio"]:
                if input_key in data:
                    model_inputs.append(data[input_key][0])

            ret_val, _ = my_model(*model_inputs)
            item = {"key": key, "value": ret_val}
            if output_path is not None:
                for suffix, wave in ret_val.items():
                    file_name = key.replace(".wav", "") + "_" + suffix + ".wav"
                    save_path = os.path.join(output_path, file_name)
                    save_audio(
                        wave[0],
                        save_path,
                        rescale=True,
                        sample_rate=my_model.codec_model.model.quantizer.sampling_rate,
                    )
            else:
                result_list.append(item)
        logging.info("inferencing is done!!")

        return result_list

    return _forward

@torch.no_grad()
def inference(args: argparse.Namespace):
    l:logging.Logger = args.logging
    args = AttrDict(**vars(args))
    os.makedirs(args.output_dir, exist_ok= True)
    ## load model
    ckpt = torch.load(args.model_file)
    model:torch.nn.Module = init(args.model)
    model.load_state_dict(ckpt['model_state_dict'])
    model.cuda()
    model.eval()
    l.info("model successfully intialized")
    ## load decoder:
    if args.decoder is not None:
        d_conf = args.decoder
        decoder = WavLMKmeansConformer(kmeans_path= d_conf['kmeans_ckpt'], 
                                    kernel_size= d_conf['kernel_size'],
                                    hifi_path  = d_conf['hifi_path'], 
                                    hifi_config= d_conf['hifi_config'])
        d_ckpt = strip_ddp_state_dict(torch.load(d_conf['conformer_ckpt'])['model_state_dict'])
        decoder.load_state_dict(d_ckpt, strict=False)
        decoder.cuda()
        decoder.eval()
    else:
        l.info("using ref decoder")
        lm = init(args.lm_model)
        film = init(args.FiLM)
        fusion = init(args.cross_attention_model)
        decoder = init(args.ref_decoder, lm_model = lm, fusion = fusion, film = film)
        decoder.cuda()
        decoder.eval()
        pass
    ## init data
    ## Convert args.data_path_and_name_and_type to a list of tuples:
    if len(args.data_path_and_name_and_type) == 1:
        args.data_path_and_name_and_type = args.data_path_and_name_and_type[0]
    else:
        res = []
        for path, name, _type in args.data_path_and_name_and_type:
            res.append((path, name, _type))
        print(res)
        args.data_path_and_name_and_type = res
    loader = Text2AudioGenTask.build_streaming_iterator(
        args.data_path_and_name_and_type,
        dtype="float32",
        batch_size=1,
        key_file=None,
        num_workers=0,
        preprocess_fn=None,
        collate_fn=Text2AudioGenTask.build_collate_fn(
            None, False, raw_sequence=("text", "prompt_text", "aux")
        ),
        allow_variable_data_keys=True,
        inference=True,
    )
    l.info("data initialized successfully")
    for keys, data in tqdm.tqdm(loader):
        torch.cuda.empty_cache()
        key = keys[0]
        logging.info(f"generating {key}")
        model_inputs = [data["text"][0], data['aux'][0]]
        # for input_key in ["prompt_text", "prompt_audio"]:
        #     if input_key in data:
        #         model_inputs.append(data[input_key][0])
        for i, e in enumerate(model_inputs):
            model_inputs[i] = torch.from_numpy(e).cuda()
        # l.info(f"model_inputs: {model_inputs}")
        # l.info(f"model_inputs shape: {model_inputs}")
        # TODO: change this in the future
        continual = model.kmeans(model_inputs[-1].unsqueeze(0)).squeeze(0).unsqueeze(-1).tolist() # list [T, 1]
        ret_val = model.decode_codec(*model_inputs, continual = continual) # [1, T, 1]
        if ret_val.size(1) ==0:
            print(f"not generating audio for ret_val {key}")
            continue
        ret_val = ret_val.squeeze(-1) # [1,T]
        l.info(f"ret_val: {ret_val.shape}")
        if args.decoder is not None:
            audio = decoder.inference(ret_val) #[1,T']
        else:
            audio = decoder.recon_audio(model.kmeans.emb(ret_val), model_inputs[-1].unsqueeze(0))
        save_path = os.path.join(args.output_dir, key+".wav")
        save_audio(audio, save_path, 16000, rescale= True)
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
