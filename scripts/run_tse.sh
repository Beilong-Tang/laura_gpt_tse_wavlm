base_path=/Netdata/2021/zb/data/LibriMix/Libri2Mix
list_path=$base_path/wav16k/min/lists
output_path=$base_path/wav16k/min/wavlm_6
num_proc=2
gpus="cuda:4 cuda:5"
wavlm_ckpt=/DKUdata/tangbl/privacy/kmeans_wavlm/ckpt/WavLM-Large.pt

mode=("test" "dev")

### Generate Libri2mix Test Dev data
for data in "${mode[@]}"; do
    echo "Processing $data"
    python -u exp/laura_gpt_se/scripts/dump_tse_data_to_wavlm.py \
        --aux_scp $list_path/$data/aux_s1.scp \
        --mix_scp $list_path/$data/mix.scp \
        --output_dir $output_path/$data \
        --wavlm_ckpt $wavlm_ckpt \
        --num_proc $num_proc \
        --gpus $gpus
done

### Generate Libi2mix Train Data
python -u exp/laura_gpt_se/scripts/dump_tse_data_to_wavlm.py \
        --aux_scp $list_path/train/all/aux_s1.scp \
        --mix_scp $list_path/train/all/mix.scp \
        --output_dir $output_path/train/all \
        --wavlm_ckpt $wavlm_ckpt \
        --num_proc $num_proc \
        --gpus $gpus