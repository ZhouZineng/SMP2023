
LR=1e-4
DATASET='DuSQL+Cspider+NL2SQL'

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

deepspeed --num_gpus=2 --master_port $MASTER_PORT main.py \
    --deepspeed deepseed.json \
    --do_train \
    --train_file ./sql_chtglm_train_final.json \
    --validation_file ./sql_chtglm_dev_final.json\
    --prompt_column query \
    --response_column answer \
    --overwrite_cache \
    --model_name_or_path /data/LLM/chatglm2-6b \
    --output_dir ./output/adgen-chatglm2-6b-ft-$LR-$DATASET \
    --preprocessing_num_workers 64 \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 256 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 800 \
    --logging_steps 10 \
    --save_steps 200 \
    --learning_rate $LR \
    --bf16

