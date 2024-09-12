import argparse
import json
import os
import pathlib
import textwrap
import time

import numpy as np
import pandas as pd
import torch
import tqdm
from IPython.display import Markdown, display
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def to_markdown(text):
    text = text.replace("•", "  *")
    return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))


prompt_fp = "./prompts/detect_category.txt"
save_fp = "./categorize_Qwen2-72B-Instruct-AWQ.json"

prompt = open(prompt_fp).read()

ct, ignore = 0, 0

model_id = "Qwen/Qwen2-72B-Instruct-AWQ"
number_gpus = 1

tokenizer = AutoTokenizer.from_pretrained(model_id)

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=32, stop=list("\n"), seed=22)
llm = LLM(model=model_id, tensor_parallel_size=number_gpus, download_dir="./save_models")

df = pd.read_csv("../CLAP_freesound/freesound_meta.csv")
path_list = os.listdir("../CLAP_freesound/freesound/test/scratch/freesound_new/test")
json_file = []
error_json = []
for index in tqdm.tqdm(range(len(df))):
    instance = {}
    tmp_file_name = df.loc[index, "audio_filename"]
    if len(tmp_file_name) > 3:
        file_path = "../CLAP_freesound/freesound/train_1/scratch/freesound_new/" + tmp_file_name.split(".")[0] + ".json"
        file_path2 = (
            "../CLAP_freesound/freesound/train_2/scratch/freesound_new/" + tmp_file_name.split(".")[0] + ".json"
        )

        try:
            with open(file_path, "r") as file:
                data = json.load(file)
        except:
            try:
                with open(file_path2, "r") as file:
                    data = json.load(file)
            except:
                continue

        tag = data["tag"]
        text = data["text"]
        description = data["original_data"]["description"]

        cur_prompt = (
            prompt.replace("{{Tag}}", str(tag))
            .replace("{{Text}}", str(text))
            .replace("{{Description}}", str(description))
        )
        instance["id"] = data["original_data"]["id"]
        instance["title"] = data["original_data"]["title"]
        instance["prompt"] = cur_prompt

        outputs = llm.generate(cur_prompt, sampling_params, use_tqdm=False)

        generated_text = outputs[0].outputs[0].text
        instance["result"] = generated_text

        json_file.append(instance)
    else:
        instance["audio_file_name"] = tmp_file_name
        error_json.append(instance)

with open(save_fp, "w") as f:
    json.dump(json_file, f, indent=4)
