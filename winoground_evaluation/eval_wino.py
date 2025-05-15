import io
import os
import re
import json
import torch
import numpy as np
import math
import shortuuid
import argparse
from PIL import Image
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList)
from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM



from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import functools

import multiprocessing as mp
from argparse import ArgumentParser
from multiprocessing import Pool


def run(rank, world_size, args):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(args.ori_processor_path) 

    model = model.to(torch.device(rank))
    model = model.eval()

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]


    ### split val
    rank = rank
    world_size = world_size

    split_length = math.ceil(len(questions)/world_size)
    logger.info("Split Chunk Length:" + str(split_length))
    split_images = questions[int(rank*split_length) : int((rank+1)*split_length)]
    logger.info(len(split_images))
      

    answers = []
    for image in tqdm(split_images): 
        image_path = args.image_folder + image['image'] + ".png"
        question = "Does the given caption accurately describe the given image? Caption:" +  image["caption"] + ". Give your answer and reasons"
            
        question2 = (
            ". Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
            "The output answer format should be as follows:\n"
            "<think> ... </think> <answer> ... </answer>\n"
            "Please strictly follow the format."
        )

        query = question# + question2
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path}
                ] + [{"type": "text", "text": query}],
            }
        ]
        

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        generation_config = GenerationConfig(
            max_new_tokens=1024,
            do_sample=True
        )
        
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, generation_config=generation_config)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = response[0]
        # logger.info(response)

        # extract answer
        content_match = re.search(r'<answer>(.*?)</answer>', response)
        response = content_match.group(1).strip() if content_match else response.strip()
        # response = '<answer>'+response+'</answer>'


        answers.append({"prompt": query,
                        "text": response})
 
    return answers

def main(args):
    multiprocess = torch.cuda.device_count() >= 2
    mp.set_start_method('spawn')
    if multiprocess:
        logger.info('started generation')
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        with Pool(world_size) as pool:
            func = functools.partial(run, world_size=world_size, args=args)
            result_lists = pool.map(func, range(world_size))

        total_results = []
        for i in range(world_size):
            total_results = total_results + result_lists[i]
            
        print(len(total_results))

        ### save path
        with open(args.answer_file, "w", encoding='utf-8') as json_file:
            for item in total_results:
                # 将字典转换为 JSON 字符串并写入文件
                json_file.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info('finished running')
    else:
        logger.info("Not enough GPUs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ori_processor_path", type=str, default="/home/data/wyy/checkpoints/Qwen2-VL-7B-Instruct")
    parser.add_argument("--model-path", type=str, default="/home/data/wyy/checkpoints/Qwen2-VL-7B-Instruct")
    parser.add_argument("--image-folder", type=str, default="/home/data/wyy/projects/SeVa/seva/playground/data/eval/winoground/data/images/")
    parser.add_argument("--question-file", type=str, default="/home/data/wyy/projects/Visual-RFT/winoground_evaluation/parsed_winoground.jsonl")
    parser.add_argument("--answer-file", type=str, default="/home/data/wyy/projects/Visual-RFT/eval_results/winoground/Qwen2_vl_7b_base.json")
    args = parser.parse_args()
    main(args)
