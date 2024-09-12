import argparse
import torch
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm    
from transformers import ClapModel
import json
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader
import ast
from utils import get_retrieval_audio_embeddings

def parse_args():
    parser = argparse.ArgumentParser()     
    parser.add_argument("--data", type=str, default="animal_sounds", required=False)    #animal_sounds, height_of_sounds
    parser.add_argument("--set", type=str, default="train", required=False)             #train, dev, test
    parser.add_argument("--retrieval_set", type=str, default="all", required=False)     #all, train, dev, test       
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

def retrieval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    if args.data == "animal_sounds":
        df = pd.read_csv(f"../animal_sounds_{args.set}.csv")
        if args.set != 'wiki':
            df['audio_path'] = df['audio_path'].apply(ast.literal_eval).apply(lambda x: x[0])
    elif args.data == "height_of_sounds":
        df = pd.read_csv(f"../height_of_sounds_{args.set}_clap.csv")

    model_path = "../models"
    model = ClapModel.from_pretrained("laion/clap-htsat-fused", cache_dir=model_path)
    model.to(device)


    audio_embeddings = get_retrieval_audio_embeddings(args.data, args.retrieval_set, model, device)

    with open(f'../text_{args.data}_{args.set}_processing.json', 'r') as f:
        test_process = json.load(f)

    class TextDataset(Dataset):
        def __init__(self, process_dict):
            self.process_dict = process_dict

        def __len__(self):
            return len(self.process_dict)

        def __getitem__(self, idx):
            text_inputs = self.process_dict[str(idx)]['text_inputs']
            return {key: torch.tensor(value) for key, value in text_inputs.items()}


    test_dataset = TextDataset(test_process)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


    audio_paths = list(audio_embeddings.keys())
    audio_embeddings_tensor = torch.tensor([item[0] for item in audio_embeddings.values()]).to(device)

    top_50_audio_paths = []

    for batch_idx, text_inputs in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        text_inputs = {key: value.squeeze(1).to(device) for key, value in text_inputs.items()}
        
        with torch.no_grad():
            text_embeddings = model.get_text_features(**text_inputs).cpu().numpy()


        similarities = cosine_similarity(text_embeddings, audio_embeddings_tensor.cpu().numpy())
        
        
        for i in range(similarities.shape[0]):
            similarity_scores = similarities[i]
            top_50_indices = similarity_scores.argsort()[-50:][::-1]

            top_50_audio_paths.append([audio_paths[idx] for idx in top_50_indices])

    top_50_df = pd.DataFrame(top_50_audio_paths, columns=[f'top_{i}_audio_path' for i in range(1, 51)])

    df = pd.concat([df, top_50_df], axis=1)

    df.to_csv(f'../retrieval_results/{args.data}_{args.set}_retrieval.csv', index=False)


if __name__ == "__main__":
    args = parse_args()
    print("----args_info----")
    print(args)
    seed_everything(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    retrieval(args)
