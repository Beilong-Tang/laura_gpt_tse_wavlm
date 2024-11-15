python -u train.py --config exp/tse_wavlm/config/conf.yaml --gpus 6,7 --log ./log/tse_wavlm

## Train using librispeech
python -u train.py --config exp/tse_wavlm/config/conf_ljspeech.yaml --gpus 6,7 --log ./log/tse_wavlm