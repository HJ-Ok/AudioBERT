import argparse
import os
import pandas as pd
from transformers import BertTokenizer, BertForTokenClassification, AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import random
from trainer import encode_data, train, evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, required=False)
    parser.add_argument("--batch_size", type=int, default=16, required=False)
    parser.add_argument("--lr", type=float, default=1e-5, required=False)
    parser.add_argument("--device", type=str, default="0", required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--train_data", type=str, default="combined", required=False)
    return parser.parse_args()

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.determinitmpic = True
    torch.backends.cudnn.benchmark = True

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    df_train_animal = pd.read_csv('../animal_sounds_train.csv')
    df_train_height = pd.read_csv('../height_of_sounds_train.csv')
    if args.train_data == "combined":
        df_train = pd.concat([df_train_animal, df_train_height])
    elif args.train_data == "animal_sounds":
        df_train = df_train_animal
    elif args.train_data == "height_of_sounds":
        df_train = df_train_height

    df_dev_animal = pd.read_csv('../animal_sounds_dev.csv')
    df_dev_height = pd.read_csv('../height_of_sounds_dev.csv')
    df_test_animal = pd.read_csv('../animal_sounds_test.csv')
    df_test_height = pd.read_csv('../height_of_sounds_test.csv')
    df_test_animal_wiki = pd.read_csv('../animal_sounds_wiki.csv')
    df_test_height_wiki = pd.read_csv('../height_of_sounds_wiki.csv')

    df_dev = pd.concat([df_dev_animal, df_dev_height])
    df_test = pd.concat([df_test_animal, df_test_height])
    df_test_wiki = pd.concat([df_test_animal_wiki, df_test_height_wiki])

    train_inputs, train_masks, train_labels = encode_data(df_train, tokenizer)

    val_inputs, val_masks, val_labels = encode_data(df_dev, tokenizer)
    val_inputs_animal, val_masks_animal, val_labels_animal = encode_data(df_dev_animal, tokenizer)
    val_inputs_height, val_masks_height, val_labels_height = encode_data(df_dev_height, tokenizer)
    test_inputs, test_masks, test_labels = encode_data(df_test, tokenizer)
    test_inputs_animal, test_masks_animal, test_labels_animal = encode_data(df_test_animal, tokenizer)
    test_inputs_height, test_masks_height, test_labels_height = encode_data(df_test_height, tokenizer)
    test_inputs_animal_wiki, test_masks_animal_wiki, test_labels_animal_wiki = encode_data(df_test_animal_wiki, tokenizer)
    test_inputs_height_wiki, test_masks_height_wiki, test_labels_height_wiki = encode_data(df_test_height_wiki, tokenizer)
    test_inputs_wiki, test_masks_wiki, test_labels_wiki = encode_data(df_test_wiki, tokenizer)

    

    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)

    val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
    val_dataset_animal = TensorDataset(val_inputs_animal, val_masks_animal, val_labels_animal)
    val_dataset_height = TensorDataset(val_inputs_height, val_masks_height, val_labels_height)
    test_dataset = TensorDataset(test_inputs, test_masks, test_labels)
    test_dataset_animal = TensorDataset(test_inputs_animal, test_masks_animal, test_labels_animal)
    test_dataset_height = TensorDataset(test_inputs_height, test_masks_height, test_labels_height)
    test_dataset_wiki = TensorDataset(test_inputs_wiki, test_masks_wiki, test_labels_wiki)
    test_dataset_animal_wiki = TensorDataset(test_inputs_animal_wiki, test_masks_animal_wiki, test_labels_animal_wiki)
    test_dataset_height_wiki = TensorDataset(test_inputs_height_wiki, test_masks_height_wiki, test_labels_height_wiki)
    
    
    

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size)

    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=args.batch_size)
    val_dataloader_animal = DataLoader(val_dataset_animal, sampler=SequentialSampler(val_dataset_animal), batch_size=args.batch_size)
    val_dataloader_height = DataLoader(val_dataset_height, sampler=SequentialSampler(val_dataset_height), batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.batch_size)
    test_dataloader_animal = DataLoader(test_dataset_animal, sampler=SequentialSampler(test_dataset_animal), batch_size=args.batch_size)
    test_dataloader_height = DataLoader(test_dataset_height, sampler=SequentialSampler(test_dataset_height), batch_size=args.batch_size)
    test_dataloader_wiki = DataLoader(test_dataset_wiki, sampler=SequentialSampler(test_dataset_wiki), batch_size=args.batch_size)
    test_dataloader_animal_wiki = DataLoader(test_dataset_animal_wiki, sampler=SequentialSampler(test_dataset_animal_wiki), batch_size=args.batch_size)
    test_dataloader_height_wiki = DataLoader(test_dataset_height_wiki, sampler=SequentialSampler(test_dataset_height_wiki), batch_size=args.batch_size)
    

    optimizer = AdamW(model.parameters(), lr=args.lr)

    train_loss = 0
    for epoch in range(args.epochs):
        train_loss += train(model, train_dataloader, optimizer, device)
    print(f"Train loss {train_loss/args.epochs}")

    _, val_f1 = evaluate(model, val_dataloader, device)
    _, test_f1 = evaluate(model, test_dataloader, device)
    _, val_f1_animal = evaluate(model, val_dataloader_animal, device)
    _, test_f1_animal = evaluate(model, test_dataloader_animal, device)
    _, val_f1_height = evaluate(model, val_dataloader_height, device)
    _, test_f1_height = evaluate(model, test_dataloader_height, device)
    _, test_f1_animal_wiki = evaluate(model, test_dataloader_animal_wiki, device)
    _, test_f1_height_wiki = evaluate(model, test_dataloader_height_wiki, device)
    _, test_f1_wiki = evaluate(model, test_dataloader_wiki, device)
    

    print(f"animal dev: {np.mean(val_f1_animal):.4f}")
    print(f"animal test: {np.mean(test_f1_animal):.4f}")
    print(f"animal wiki test: {np.mean(test_f1_animal_wiki):.4f}")

    print('-'*60)

    print(f"height dev: {np.mean(val_f1_height):.4f}")
    print(f"height test: {np.mean(test_f1_height):.4f}")
    print(f"height wiki test: {np.mean(test_f1_height_wiki):.4f}")

    print('-'*60)

    print(f"combined dev: {np.mean(val_f1):.4f}")
    print(f"combined test: {np.mean(test_f1):.4f}")
    print(f'combined wiki test: {np.mean(test_f1_wiki):.4f}')

    print('-'*60)

    # model.save_pretrained("../detection_bert/detection_bert_model_combined")
    # tokenizer.save_pretrained("../detection_bert/detection_bert_tokenizer_combined")

    
    
    


if __name__ == "__main__":
    args = parse_args()
    print("----args_info----")
    print(args)
    seed_everything(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    train(args)
