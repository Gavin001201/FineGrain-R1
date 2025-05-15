import argparse
import os
import json
from tqdm import tqdm
import sys
import random
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

    # answers_file = os.path.expanduser(args.result_file)
    # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # ans_file = open(answers_file, "w")

    # 初始统计
    correct_count = defaultdict(int)
    total_count = defaultdict(int)

    # 数据处理
    for i in range(0, len(data_list), 2):
        first_entry = data_list[i]
        second_entry = data_list[i + 1]

        category = "_".join(first_entry["question_id"].split('_')[:-1])

        # 增加总数
        total_count[category] += 1

        # 检查答案是否正确
        first_answer = format_text(first_entry["text"])
        second_answer = format_text(second_entry["text"])
        if first_answer == "A" and second_answer == "B":
            correct_count[category] += 1

    # 统计正确率
    accuracy = {category: (correct_count[category] / total_count[category]) * 100 if total_count[category] > 0 else 0 for category in total_count}

    # 输出结果
    total = 0
    num = 0
    for category, acc in accuracy.items():
        num += 1
        total += acc
        print(f"{category:<20} Accuracy: {acc:>8.2f}%")
    print(f"Avg: {(total / num):>8.2f}%")
        

    # ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, default="/home/data/wyy/projects/Visual-RFT/eval_results/sugarcrepe/Qwen2_vl_7b.json")
    parser.add_argument("--result-file", type=str, default="answer.jsonl")
    args = parser.parse_args()

    eval(args)
