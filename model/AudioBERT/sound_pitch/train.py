import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from dataloader import create_data_loader
from model import AK_BERT
from trainer import train_epoch, validate
from transformers import AutoTokenizer, ASTFeatureExtractor
from transformers.optimization import get_cosine_schedule_with_warmup
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")


def peft_state_dict(model: nn.Module, bias: str = "none"):
    my_state_dict = model.state_dict()
    if bias == "none":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k}
    elif bias == "all":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k or "bias" in k}
    elif bias == "LoRA" or bias == "AdaLoRA":
        to_return = {}
        for k in my_state_dict:
            if "lora_" in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    elif bias == "IA3":
        to_return = {}
        for k in my_state_dict:
            if "ia3_l" in k:
                to_return[k] = my_state_dict[k]
        return to_return
    else:
        raise NotImplementedError


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20, required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--lr", type=float, default=3e-4, required=False)
    parser.add_argument("--device", type=str, default="0", required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--data_path", type=str, default="./", required=False)
    parser.add_argument("--output_path", type=str, default="./outputs", required=False)
    parser.add_argument("--language_model_name", type=str, default="google-bert/bert-base-uncased", required=False)
    parser.add_argument(
        "--audio_model_name", type=str, default="MIT/ast-finetuned-audioset-10-10-0.4593", required=False
    )
    parser.add_argument("--save_model_name", type=str, default="audio-bert", required=False)
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.determinitmpic = True
    torch.backends.cudnn.benchmark = True


def Train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv("./height_of_sounds_train.csv")
    df2 = pd.read_csv("./height_of_sounds_dev.csv")

    train_df = pd.read_csv(
        "./datasets/base_model/height_of_sounds_train_top_50_audio_path_height_train_height_base.csv"
    )
    dev_df = pd.read_csv("./datasets/base_model/height_of_sounds_dev_top_50_audio_path_height_dev_height_base.csv")

    train_df = train_df.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
    df["id_1_audio_path"] = df["id_1"].map(train_df.set_index("id")["top_1_audio_path"])
    df["id_2_audio_path"] = df["id_2"].map(train_df.set_index("id")["top_1_audio_path"])
    train_df = df

    dev_df = dev_df.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
    df2["id_1_audio_path"] = df2["id_1"].map(dev_df.set_index("id")["top_1_audio_path"])
    df2["id_2_audio_path"] = df2["id_2"].map(dev_df.set_index("id")["top_1_audio_path"])
    dev_df = df2

    tokenizer = AutoTokenizer.from_pretrained(args.language_model_name)
    extractor = ASTFeatureExtractor.from_pretrained(args.audio_model_name)

    train_data_loader = create_data_loader(train_df, tokenizer, extractor, args.batch_size, shuffle_=True)
    dev_data_loader = create_data_loader(dev_df, tokenizer, extractor, args.batch_size)

    model = AK_BERT(
        language_model_path=args.language_model_name,
        audio_model_path=args.audio_model_name,
        tokenizer=tokenizer,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_data_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    max_acc = 0
    for epoch in range(args.epochs):
        print("-" * 10)
        print(f"Epoch {epoch}/{args.epochs-1}")
        print("-" * 10)
        train_acc, train_loss = train_epoch(model, train_data_loader, optimizer, device, scheduler, epoch)
        _, dev_loss, dev_acc = validate(model, dev_data_loader, device)

        if dev_acc > max_acc:
            max_acc = dev_acc
            torch.save(
                peft_state_dict(model.language_enc, bias="LoRA"),
                os.path.join("./weights", f"{args.save_model_name}_{args.seed}_LORA.pt"),
            )

        print(f"Train loss {train_loss} accuracy {train_acc}")
        print(f"Dev loss {dev_loss} accuracy {dev_acc}")
        print("")

    print(f"Best dev acc {max_acc}")


if __name__ == "__main__":
    args = parse_args()
    print("----args_info----")
    print(args)
    seed_everything(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    Train(args)
