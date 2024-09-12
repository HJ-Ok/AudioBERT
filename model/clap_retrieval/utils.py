import torch
import pandas as pd
import json
from torch.utils.data import DataLoader, Dataset

class AudioDataset(Dataset):
    def __init__(self, process):
        self.process = process

    def __len__(self):
        return len(self.process)

    def __getitem__(self, idx):
        audio_input = self.process[str(idx)]['inputs']
        audio_path = self.process[str(idx)]['audio_path']
        return audio_input, audio_path

def detect_target_word(sentence, tokenizer, model, device):
    encoding = tokenizer(sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    predictions = predictions.squeeze().tolist()

    detected_texts = []
    current_sentence = ""
    current_word = ""

    for token, label in zip(tokens, predictions):
        if label == 1:
            if token.startswith("##"):
                current_word += token[2:]
            else:
                if current_word:
                    current_sentence += current_word + " "
                current_word = token
        else:
            if current_word:
                current_sentence += current_word + " "
                current_word = ""
            
            if current_sentence:
                detected_texts.append(current_sentence.strip().replace(" - ", "-"))
                current_sentence = ""

    if current_word:
        current_sentence += current_word
        detected_texts.append(current_sentence.strip().replace(" - ", "-"))


    return detected_texts

def get_retrieval_audio_embeddings(data, set, model, device):
    
    if data == 'height_of_sounds':
        df_train = pd.read_csv(f'/workspace/shyoo/data_v2/{data}_train_clap.csv')
        df_test = pd.read_csv(f'/workspace/shyoo/data_v2/{data}_test_clap.csv')
    else:
        df_train = pd.read_csv(f'/workspace/shyoo/data_v2/{data}_train.csv')
        df_test = pd.read_csv(f'/workspace/shyoo/data_v2/{data}_test.csv')

    audio_process_path = f'/workspace/shyoo/data_v2/audio_{data}_processing.json' 
    with open(audio_process_path, 'r') as f:
        process = json.load(f)

    if set == 'all':
        for idx in process:
            process[idx] = {
                "inputs": {key: torch.tensor(value) for key, value in process[idx]['inputs'].items()},
                "audio_path": process[idx]["audio_path"]
            }
        cur_process = process
    if set == 'train':
        cur_process = {}
        for idx in process:
            if int(idx) < len(df_train):
                cur_process[idx] = {
                    "inputs": {key: torch.tensor(value) for key, value in process[idx]['inputs'].items()},
                    "audio_path": process[idx]["audio_path"]
                }
    if set == 'test':
        cur_process = {}
        for idx in process:
            if int(idx) >= len(df_train) and int(idx) < len(df_train) + len(df_test):
                cur_process[str(int(idx) - len(df_train))] = {
                    "inputs": {key: torch.tensor(value) for key, value in process[idx]['inputs'].items()},
                    "audio_path": process[idx]["audio_path"]
                }
    if set == 'dev':
        cur_process = {}
        for idx in process:
            if  int(idx) >= len(df_train) + len(df_test):
                cur_process[str(int(idx) - len(df_train) - len(df_test))] = {
                    "inputs": {key: torch.tensor(value) for key, value in process[idx]['inputs'].items()},
                    "audio_path": process[idx]["audio_path"]
                }
        

    dataset = AudioDataset(cur_process)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)

    audio_embeddings = {}
    for batch in dataloader:
        audio_inputs, audio_paths = batch
        
        audio_inputs = {key: value.squeeze(1).to(device) for key, value in audio_inputs.items()}
        
        with torch.no_grad():
            audio_embedding = model.get_audio_features(**audio_inputs)

        for i in range(audio_embedding.size(0)):
            audio_embeddings[audio_paths[i]] = (audio_embedding[i].cpu().tolist(), audio_paths[i])

    return audio_embeddings