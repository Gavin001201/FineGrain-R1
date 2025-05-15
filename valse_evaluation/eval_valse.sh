#!/bin/bash
ANSWER_FILE=/home/data/wyy/projects/Visual-RFT/valse_evaluation/7b-sft-prompt-explore.json

python ./valse_evaluation.py \
    --ori_processor_path /home/data/wyy/checkpoints/Qwen2-VL-7B-Instruct \
    --model-path /home/data/wyy/projects/Qwen2-VL-Finetune/output/7b-sft/checkpoint-78 \
    --answer-file ${ANSWER_FILE} \

python ./eval.py \
    --annotation-file ${ANSWER_FILE} \