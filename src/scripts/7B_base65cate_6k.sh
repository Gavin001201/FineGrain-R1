export DATA_PATH=/home/data/wyy/projects/Visual-RFT/dataset/spatial_dataset_10k_8
export CKPT_PATH=/home/data/wyy/checkpoints/Qwen2-VL-7B-Instruct
export SAVE_PATH=/home/data/wyy/projects/Visual-RFT/checkpoints/Qwen2-VL-7B-spatial_10k_normalize_epoch_1_4

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="/home/data/wyy/projects/Visual-RFT/debug_log_7b_GRPO_spatial_10k_normalize_8_epoch_1_4.txt"

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --deepspeed ./local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing False \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name Qwen2-VL-7B_GRPO_spatial_10k_normalize_8_epoch_1_4 \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 4
