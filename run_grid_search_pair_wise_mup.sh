warmup_ratio=0.01
exit_steps=20000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,
for output_mult in 5
do
    for lr in 5e-5 1e-4 3e-4 1e-3 3e-3
    do
        for initializer_range in 0.05
        do
            for hp_tune_actual_width in 256 512
            do
                python -m torch.distributed.launch --nproc_per_node=8 \
                --nnodes=1 \
                run_train_gpt_mup_from_scratch.py \
                --model_name_or_path gpt2 \
                --model_load_pretrained False \
                --config_name ./configs/gpt_2_L_6 \
                --output_dir ./res/output/pair_wise_L6/${lr}_${output_mult}_${initializer_range}/width_${hp_tune_actual_width} \
                --final_train_dir "/path/to/your/data" \
                --overwrite_output_dir \
                --num_train_epochs 0.1 \
                --per_device_train_batch_size 6 \
                --warmup_ratio ${warmup_ratio} \
                --ddp_timeout 1000000 \
                --logging_steps 100 \
                --save_steps 5000 \
                --save_total_limit 20 \
                --learning_rate ${lr} \
                --hp_tune_base_width 256 \
                --size_per_head 64 \
                --hp_tune_actual_width ${hp_tune_actual_width} \
                --output_mult ${output_mult} \
                --initializer_range ${initializer_range} \
                --log_warmup True \
                --unified_dropout 0.0 \
                --exit_steps ${exit_steps} \
                --deepspeed ./deepspeed_configs/zero_stage1_config.json \
                > logs/pair_wise_L6_test_release_${lr}_${output_mult}_${initializer_range}_width_${hp_tune_actual_width}.txt 2>&1

                sleep 60
            done
        done
    done
done