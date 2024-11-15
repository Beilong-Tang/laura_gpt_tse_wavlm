python -u inference/infer.py --config_file exp/tse_wavlm/config/conf.yaml --model_file /DKUdata/tangbl/laura_gpt_tse/exp/tse_wavlm/ckpt/conf/best.pth --output_dir /DKUdata/tangbl/laura_gpt_tse/output/conf_no_ref --default_config inference/infer_no_ref_decoder.yaml --device cuda:3

## Infer with auxilary audio
python -u inference/infer.py --config_file exp/tse_wavlm/config/conf.yaml --model_file /DKUdata/tangbl/laura_gpt_tse/exp/tse_wavlm/ckpt/conf/best.pth --output_dir /DKUdata/tangbl/laura_gpt_tse/output/conf --default_config inference/infer.yaml --device cuda:2