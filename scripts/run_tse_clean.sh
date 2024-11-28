base_path=/Netdata/2021/zb/data/LibriMix/Libri2Mix
list_path=$base_path/wav16k/min/lists
output_path=$base_path/wav16k/min/wavlm_6
num_proc=2
gpus="cuda:6 cuda:7"
wavlm_ckpt=/DKUdata/tangbl/privacy/kmeans_wavlm/ckpt/WavLM-Large.pt

mode=("test" "dev")

### Generate Libri2mix Testing and Dev Data
for data in "${mode[@]}"; do
    echo "Processing $data"
    python -u exp/laura_gpt_se/scripts/dump_data_to_wavlm.py --clean_scp $list_path/$data/s1.scp \
        --mix_scp "" \
        --output_dir $output_path/$data \
        --wavlm_ckpt $wavlm_ckpt \
        --num_proc $num_proc \
        --gpus $gpus \
        --only_clean 
done

### Generate Training Data
python -u exp/laura_gpt_se/scripts/dump_data_to_wavlm.py --clean_scp $list_path/train/all/s1.scp \
        --mix_scp "" \
        --output_dir $output_path/train/all \
        --wavlm_ckpt $wavlm_ckpt \
        --num_proc $num_proc \
        --gpus $gpus \
        --only_clean 