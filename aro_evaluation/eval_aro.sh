#!/bin/bash
MODEL_VERSION=7b-sft-simple-prompt
MODEL_PATH=/home/data/wyy/projects/Qwen2-VL-Finetune/output/7b-sft/checkpoint-78

# python ./aro.py \
#     --ori_processor_path /home/data/wyy/checkpoints/Qwen2-VL-7B-Instruct \
#     --model-path ${MODEL_PATH} \
#     --image-folder /home/data/wyy/datasets/vg/VG_100K \
#     --question-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/aro/parsed_vg_att.jsonl \
#     --answer-file /home/data/wyy/projects/Visual-RFT/eval_results/aro/${MODEL_VERSION}/vg_att.jsonl

# python ./aro.py \
#     --ori_processor_path /home/data/wyy/checkpoints/Qwen2-VL-7B-Instruct \
#     --model-path ${MODEL_PATH} \
#     --image-folder /home/data/wyy/datasets/vg/VG_100K \
#     --question-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/aro/parsed_vg_rel.jsonl \
#     --answer-file /home/data/wyy/projects/Visual-RFT/eval_results/aro/${MODEL_VERSION}/vg_rel.jsonl

python ./eval.py \
    --annotation-file /home/data/wyy/projects/Visual-RFT/eval_results/aro/${MODEL_VERSION}/vg_att.jsonl \

# python ./eval.py \
#     --annotation-file /home/data/wyy/projects/Visual-RFT/eval_results/aro/${MODEL_VERSION}/vg_rel.jsonl \