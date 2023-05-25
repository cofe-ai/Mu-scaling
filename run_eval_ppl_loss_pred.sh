# WikiText2 : wikitext & wikitext-2-raw-v1
CUDA_VISIBLE_DEVICES=1
#export HF_DATASETS_OFFLINE=1
params="0.06_2e-3_4"
for width in 128 256 384 1024
do
    python run_eval_ppl_mup.py \
    --cache_dir /your/huggingface/cache/dir \
    --is_ours \
    --dataset_path wikitext \
    --dataset_name wikitext-2-raw-v1 \
    --model_name_or_path res/output/test_standard_mup_loss_pred_20k/${params}/width_${width}/checkpoint-20000 \
    > logs/eval/20k_${params}_${width}.txt 2>&1
done