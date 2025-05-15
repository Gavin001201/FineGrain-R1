#!/bin/bash
# MODEL_VERSION=Qwen2_vl_2b_base_spatial10k_8_epoch_1_simple_prompt
MODEL_VERSION=7b-sft-prompt-explore
MODEL_PATH=/home/data/wyy/projects/Qwen2-VL-Finetune/output/7b-sft/checkpoint-78

python ./eval_wino.py \
    --ori_processor_path /home/data/wyy/checkpoints/Qwen2-VL-7B-Instruct \
    --model-path ${MODEL_PATH} \
    --image-folder /home/data/wyy/projects/SeVa/seva/playground/data/eval/winoground/data/images/ \
    --question-file /home/data/wyy/projects/Visual-RFT/winoground_evaluation/parsed_winoground.jsonl \
    --answer-file /home/data/wyy/projects/Visual-RFT/eval_results/winoground/${MODEL_VERSION}.json

# python ./eval.py \
#     --answer-file /home/data/wyy/projects/Visual-RFT/eval_results/winoground/${MODEL_VERSION}.json \
#     --gpt-log /home/data/wyy/projects/Visual-RFT/eval_results/winoground/${MODEL_VERSION}_gpt.json