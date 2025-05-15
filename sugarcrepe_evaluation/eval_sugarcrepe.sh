#!/bin/bash
ANSWER_FILE=/home/data/wyy/projects/Visual-RFT/eval_results/sugarcrepe/7b-sft.json

# python ./sugar.py \
#     --ori_processor_path /home/data/wyy/checkpoints/Qwen2-VL-7B-Instruct \
#     --model-path /home/data/wyy/projects/Qwen2-VL-Finetune/output/7b-sft/checkpoint-78 \
#     --image-folder /home/data/wyy/datasets/coco2017/val2017 \
#     --question-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/sugarcrepe/parsed_sugar.jsonl \
#     --answer-file ${ANSWER_FILE} \

python ./eval.py \
    --annotation-file ${ANSWER_FILE} \