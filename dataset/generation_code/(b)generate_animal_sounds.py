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

with open("./categorize_Qwen2-72B-Instruct-AWQ.json", "r") as file:
    data = json.load(file)

for i in data:
    id = i["id"]
    result = i["result"]

    if not result.startswith("animal"):
        continue
    df.loc[df["freesound_id"] == str(id), "category"] = result


def to_markdown(text):
    text = text.replace("•", "  *")
    return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))


prompt_fp = "./prompts/animal_sounds.txt"
save_fp = ".animal_sound_Qwen2-72B-Instruct-AWQ_seed42.json"


prompt = open(prompt_fp).read()

ct, ignore = 0, 0

model_id = "Qwen/Qwen2-72B-Instruct-AWQ"
number_gpus = 1

tokenizer = AutoTokenizer.from_pretrained(model_id)

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=100, seed=42)
llm = LLM(model=model_id, tensor_parallel_size=number_gpus, download_dir="./save_models")

path_list = os.listdir("../CLAP_freesound/freesound/test/scratch/freesound_new/test")

json_file = []
for index in tqdm.tqdm(range(len(df))):
    instance = {}
    tmp_file_name = df.loc[index, "audio_filename"]
    file_path = "../CLAP_freesound/freesound/train_1/scratch/freesound_new/" + tmp_file_name.split(".")[0] + ".json"
    file_path2 = "../CLAP_freesound/freesound/train_2/scratch/freesound_new/" + tmp_file_name.split(".")[0] + ".json"

    try:
        with open(file_path, "r") as file:
            data = json.load(file)
    except:
        with open(file_path2, "r") as file:
            data = json.load(file)

    tag = data["tag"]
    text = data["text"]
    category = df.loc[index, "category"]
    description = data["original_data"]["description"]

    cur_prompt = (
        prompt.replace("{{Tag}}", str(tag))
        .replace("{{Text}}", str(text))
        .replace("{{Description}}", str(description))
        .replace("{{Category}}", str(category))
    )
    instance["id"] = data["original_data"]["id"]
    instance["title"] = data["original_data"]["title"]
    instance["prompt"] = cur_prompt

    outputs = llm.generate(cur_prompt, sampling_params, use_tqdm=False)

    generated_text = outputs[0].outputs[0].text
    instance["result"] = generated_text

    json_file.append(instance)


with open(save_fp, "w") as f:
    json.dump(json_file, f, indent=4)
