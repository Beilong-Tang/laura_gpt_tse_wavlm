# LauraGPT WavLM

[![Demo](https://img.shields.io/badge/Demo-green?&logo=youtube)](https://beilong-tang.github.io/LauraGPT_WavLM_TSE.demo/)

:warning: This repository is under construction, and it now contains my personal code for operating on my server. I will clean it up after finishing my experiments. But it should contain the key codes for building a LauraGPT TSE Model now.

This repository contains the lauraGPT model for TSE and SE using WavLM Features. 

To check the model detail, please visit [here](https://beilong-tang.github.io/LauraGPT_WavLM_TSE.demo/)

## Pre-requisite
1. Install [Funcodec](https://github.com/modelscope/FunCodec).
2. Follow `scripts/run.sh` to extract WavLM Features for SE tasks.
3. Follow `scripts/run_tse.sh` to extract WavLM Features for TSE tasks.

## Run

The model for SE is in `exp/se_wavlm` and the model for TSE is in `exp/tse_wavlm` 

### Speech Enhancement
To run a __SE__ task using default config, do

```shell
python -u train.py --config exp/se_wavlm/config/conf.yaml --gpus 0,1,2,3 --log log/se_wavlm/conf
```

### Target Speaker Extraction
To run a __TSE__ task using default config, do

```shell
python -u train.py --config exp/tse_wavlm/config/conf.yaml --gpus 0,1,2,3 --log ./log/tse_wavlm
```

