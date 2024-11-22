## Infer wer (ml_framework_slurm)
python recipes/whisper/main.py -t /DKUdata/tangbl/laura_gpt_tse/output/conf_no_ref -o /DKUdata/tangbl/laura_gpt_tse/output/wer/conf_no_ref/wer.txt -d cuda:2

## Infer Libri2Mix Clean wer
python recipes/whisper/main.py -t /Netdata/2021/zb/data/LibriMix/Libri2Mix/wav16k/min/test/s1 -o /DKUdata/tangbl/laura_gpt_tse/output/wer/conf_no_ref/libri_test_s1.txt -d cuda:2

## Compare (ml_framework_slurm)
python recipes/wer.py --output /DKUdata/tangbl/laura_gpt_tse/output/wer/conf_no_ref/wer.txt --reference /DKUdata/tangbl/laura_gpt_tse/output/wer/conf_no_ref/libri_test_s1.txt

## Compare wer with LibriSpeech discrete Tokens
python recipes/wer.py --output /DKUdata/tangbl/laura_gpt_tse/output/wer/conf_no_ref/wer.txt --reference /DKUdata/tangbl/ml_framework_slurm/exp/conformer/conformer/output/sb_librispeech_k_1000_transcript.txt 