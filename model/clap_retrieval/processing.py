import argparse
import torch
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoProcessor
import json
from transformers import BertForTokenClassification
from transformers import BertTokenizer
import librosa
import ast
from utils import detect_target_word


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="text", required=False)         #text, audio
    parser.add_argument("--data", type=str, default="combined", required=False)     #animal_sounds, height_of_sounds, combined
    parser.add_argument("--set", type=str, default="train", required=False)         #train, dev, test, wiki
    parser.add_argument("--device", type=str, default="0", required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    return parser.parse_args()

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.determinitmpic = True
    torch.backends.cudnn.benchmark = True

def processing(args):

    model_path = "/workspace/shyoo/models"
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused", cache_dir=model_path)

    if args.mode == "text":
        df_animal = pd.read_csv(f"/workspace/shyoo/data_v2/animal_sounds_{args.set}.csv")
        if args.set != 'wiki':
            df_animal['audio_path'] = df_animal['audio_path'].apply(ast.literal_eval).apply(lambda x: x[0])
        df_height = pd.read_csv(f"/workspace/shyoo/data_v2/height_of_sounds_{args.set}_clap.csv")
        if args.data == "combined":
            df = pd.concat([df_height, df_animal], axis=0)
        elif args.data == "animal_sounds":
            df = df_animal
        elif args.data == "height_of_sounds":
            df = df_height

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = BertForTokenClassification.from_pretrained('/workspace/shyoo/models/detection_bert/detection_bert_model_combined', num_labels=2).to(device)
        tokenizer = BertTokenizer.from_pretrained('/workspace/shyoo/models/detection_bert/detection_bert_tokenizer_combined')

        descriptions = [detect_target_word(sentence, tokenizer, model, device) for sentence in tqdm(df['sentence'])]

        dic = {}
        for idx in tqdm(range(len(df)), total=len(df)):

            max_len = 64
            if idx % 2 == 0:
                try:
                    text_inputs = processor(
                            text=f'{descriptions[idx][0]} sound',
                            return_tensors='pt',
                            padding='max_length',
                            max_length=max_len
                        )
                except:
                    text_inputs = processor(
                            text=f' sound',
                            return_tensors='pt',
                            padding='max_length',
                            max_length=max_len
                        )
            else:
                try:
                    text_inputs = processor(
                            text=f'{descriptions[idx][1]} sound',
                            return_tensors='pt',
                            padding='max_length',
                            max_length=max_len
                        )
                except:
                    try:
                        text_inputs = processor(
                            text=f'{descriptions[idx][0]} sound',
                            return_tensors='pt',
                            padding='max_length',
                            max_length=max_len
                        )
                    except:
                        text_inputs = processor(
                            text=f' sound',
                            return_tensors='pt',
                            padding='max_length',
                            max_length=max_len
                        )


            dic[idx] = text_inputs
        for idx in dic:
            dic[idx] = {
                "text_inputs": {key: value.tolist() for key, value in dic[idx].items()}
            }
            
    elif args.mode == "audio":
        if args.data == "animal_sounds":
            df_train = pd.read_csv(f"/workspace/shyoo/data_v2/animal_sounds_train.csv")        
            df_test = pd.read_csv(f"/workspace/shyoo/data_v2/animal_sounds_test.csv")
            df_dev = pd.read_csv(f"/workspace/shyoo/data_v2/animal_sounds_dev.csv")
        elif args.data == "height_of_sounds":
            df_train = pd.read_csv(f"/workspace/shyoo/data_v2/height_of_sounds_train.csv")        
            df_test = pd.read_csv(f"/workspace/shyoo/data_v2/height_of_sounds_test.csv")
            df_dev = pd.read_csv(f"/workspace/shyoo/data_v2/height_of_sounds_dev.csv")

        df = pd.concat([df_train, df_test, df_dev])
        if args.data == "animal_sounds":
            df['audio_path'] = df['audio_path'].apply(ast.literal_eval).map(lambda x: x[0])

        dic = {}
        for idx in tqdm(range(len(df)), total=len(df)):
            row = df.iloc[idx]
            audio_path = row['audio_path']
            
            audio, sr = librosa.load(audio_path, sr=None)
                
            max_len = 64
            inputs = processor(audios=[audio], sampling_rate=sr, return_tensors='pt', padding='max_length', max_length=max_len)

            dic[idx] = (inputs, audio_path)

            
        for idx in dic:
            dic[idx] = {"inputs": {key: value.tolist() for key, value in dic[idx][0].items()},
                        "audio_path": dic[idx][1]}

    with open(f'/workspace/shyoo/data_v2/{args.mode}_{args.data}_{args.set}_processing.json', 'w') as f:
            json.dump(dic, f)

if __name__ == "__main__":
    args = parse_args()
    print("----args_info----")
    print(args)
    seed_everything(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    processing(args)