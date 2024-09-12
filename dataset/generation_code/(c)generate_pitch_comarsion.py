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


prompt_fp = "./prompts/sound_pitch_comparetxt"
prompt = open(prompt_fp).read()

ct, ignore = 0, 0

model_id = "Qwen/Qwen2-72B-Instruct-AWQ"
number_gpus = 1

tokenizer = AutoTokenizer.from_pretrained(model_id)

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=100, seed=42)
llm = LLM(model=model_id, tensor_parallel_size=number_gpus, download_dir="./save_models")
path_list = os.listdir("../CLAP_freesound/freesound/test/scratch/freesound_new/test")


def get_random_pairs(df, column_name, max_pairs=50000):
    unique_ids = list(df[column_name].unique())
    pairs = []

    while len(pairs) < max_pairs and len(unique_ids) >= 2:
        pair = random.sample(unique_ids, 2)
        category_1 = df[df[column_name] == pair[0]]["category"].values[0]
        category_2 = df[df[column_name] == pair[1]]["category"].values[0]
        pairs.append(pair + [category_1, category_2])
        unique_ids.remove(pair[0])
        unique_ids.remove(pair[1])

    return pd.DataFrame(pairs, columns=[f"{column_name}_1", f"{column_name}_2", "category_1", "category_2"])


with open("./categorize_Qwen2-72B-Instruct-AWQ.json", "r") as file:
    data = json.load(file)


def update_category_by_string(data, df, category_str):
    tmp_df = df.copy()
    for i in data:
        id = i["id"]
        result = i["result"]

        if not result.startswith(category_str):
            continue
        tmp_df.loc[df_music["freesound_id"] == str(id), "category"] = result
    tmp_df_pairs = get_random_pairs(tmp_df, "audio_filename")

    return tmp_df_pairs


df = pd.read_csv("../CLAP_freesound/freesound_meta.csv")
df_music_pairs = update_category_by_string(data, df, "music")
df_object_pairs = update_category_by_string(data, df, "object")
df_environment_pairs = update_category_by_string(data, df, "environment")


def make_data(df, save_fp):
    json_file = []
    for index in tqdm.tqdm(range(len(df))):
        instance = {}
        tmp_file_name1 = df.loc[index, "audio_filename_1"]
        file_path = (
            "../CLAP_freesound/freesound/train_1/scratch/freesound_new/" + tmp_file_name1.split(".")[0] + ".json"
        )
        file_path2 = (
            "../CLAP_freesound/freesound/train_2/scratch/freesound_new/" + tmp_file_name1.split(".")[0] + ".json"
        )

        try:
            with open(file_path, "r") as file:
                data = json.load(file)
        except:
            with open(file_path2, "r") as file:
                data = json.load(file)

        tag_1 = data["tag"]
        text_1 = data["text"]
        category_1 = df.loc[index, "category_1"]
        description_1 = data["original_data"]["description"]

        tmp_file_name2 = df.loc[index, "audio_filename_2"]
        file_path = (
            "../CLAP_freesound/freesound/train_1/scratch/freesound_new/" + tmp_file_name2.split(".")[0] + ".json"
        )
        file_path2 = (
            "../CLAP_freesound/freesound/train_2/scratch/freesound_new/" + tmp_file_name2.split(".")[0] + ".json"
        )

        try:
            with open(file_path, "r") as file:
                data2 = json.load(file)
        except:
            with open(file_path2, "r") as file:
                data2 = json.load(file)

        tag_2 = data2["tag"]
        text_2 = data2["text"]
        category_2 = df.loc[index, "category_2"]
        description_2 = data2["original_data"]["description"]

        cur_prompt = (
            prompt.replace("{{Tag_1}}", str(tag_1))
            .replace("{{Text_1}}", str(text_1))
            .replace("{{Description_1}}", str(description_1))
            .replace("{{Category_1}}", str(category_1))
            .replace("{{Tag_2}}", str(tag_2))
            .replace("{{Text_2}}", str(text_2))
            .replace("{{Description_2}}", str(description_2))
            .replace("{{Category_2}}", str(category_2))
        )

        instance["id_1"] = data["original_data"]["id"]
        instance["title_1"] = data["original_data"]["title"]
        instance["id_2"] = data2["original_data"]["id"]
        instance["title_2"] = data2["original_data"]["title"]

        instance["prompt"] = cur_prompt

        outputs = llm.generate(cur_prompt, sampling_params, use_tqdm=False)

        generated_text = outputs[0].outputs[0].text
        instance["result"] = generated_text

        json_file.append(instance)

    with open(save_fp, "w") as f:
        json.dump(json_file, f, indent=4)


make_data(df_music_pairs, "sound_pitch_comparsion_music.json")
make_data(df_object_pairs, "sound_pitch_comparsion_object.json")
make_data(df_environment_pairs, "sound_pitch_comparsion_environment.json")
