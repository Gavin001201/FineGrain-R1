import argparse
import os
import json
from tqdm import tqdm
import random
import sys
sys.path.append('/home/data/wyy/projects/SeVa/seva')
from collections import defaultdict


disable_answers = []

def format_text(text):
    # 如果结果比较短，大概率格式符合要求，简单去除空格和句点，以满足比较要求
    if len(text) < 10:
        text = text.replace(' ', '').replace('.', '')
    else:
        # disable_answers.append(text)
        if 'A.' in text and 'B.' not in text:
            text = 'A'
        elif 'B.' in text and 'A.' not in text:
            text = 'B'
        elif 'Description A' in text and 'Description B' not in text:
            text = 'A'
        elif 'Description B' in text and 'Description A' not in text:
            text = 'B'
        elif '<answer>' in text:
            text = text.split('<answer>')[1].replace(' ', '').replace('.', '')
            if text not in ['A', 'B']:
                pass            
        else:
            disable_answers.append(text)
            ######TODO######
            text = random.choice(['A', 'B'])
    return text

def eval(args):
    data_list = [json.loads(q) for q in open(os.path.expanduser(args.annotation_file), "r")]

    # 初始统计
    correct_count = defaultdict(int)
    total_count = defaultdict(int)
    attributes_count = defaultdict(int)
    attributes_true_count = defaultdict(int)

    # 数据处理
    for i in range(0, len(data_list), 2):
        first_entry = data_list[i]
        second_entry = data_list[i + 1]

        category = "_".join(first_entry["question_id"].split('_')[:-1])

        if category == "vg_att":
            attribute = first_entry["attributes"]
        else:
            attribute = first_entry["relations"]
        attributes_count[attribute] += 1
        total_count[category] += 1

        # 检查答案是否正确
        first_answer = format_text(first_entry["text"])
        second_answer = format_text(second_entry["text"])
        if first_answer == "A" and second_answer == "B":
            attributes_true_count[attribute] += 1
            
        # # 检查答案是否正确
        # if first_entry["text"] == "A" and second_entry["text"] == "B":
        #     attributes_true_count[attribute] += 1
            
    # 统计正确率
    accuracy_list = []
    total_true = 0
    for attribute, true_count in attributes_true_count.items():
        total_true += true_count
        
        if category == "vg_att" and true_count >= 25:
            accuracy_list.append((true_count / attributes_count[attribute]) * 100)
        elif category == "vg_rel" and true_count > 0:
            accuracy_list.append((true_count / attributes_count[attribute]) * 100)
    
    accuracy = sum(accuracy_list) / len(accuracy_list) if accuracy_list else 0
    absolute_accuracy = total_true / total_count[category] * 100

    # 输出结果
    print(f"{category:<20} Accuracy: {accuracy:>8.2f}%")
    print(f"{category:<20} Absolute_Accuracy: {absolute_accuracy:>8.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, default="/home/data/wyy/projects/Visual-RFT/eval_results/aro/Qwen2-VL-7B/vg_att.jsonl")
    parser.add_argument("--result-file", type=str, default="answer.jsonl")
    args = parser.parse_args()

    eval(args)
