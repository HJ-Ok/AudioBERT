import pandas as pd
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

def encode_data(data, tokenizer):
    input_ids = []
    attention_masks = []
    labels = []

    for _, row in data.iterrows():
        sentence = row['sentence']
        if 'description' not in row or pd.isna(row['description']):
            target_words = [row['span1'], row['span2']]
        else:
            target_words = [row['description']]

        encoding = tokenizer(sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        input_id = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        label = [0] * len(input_id)

        for target_word in target_words:
            if pd.isna(target_word):
                continue
            target_word_tokens = tokenizer.tokenize(target_word)
            target_word_ids = tokenizer.convert_tokens_to_ids(target_word_tokens)

            for i in range(len(input_id) - len(target_word_ids) + 1):
                if input_id[i:i+len(target_word_ids)].tolist() == target_word_ids:
                    for j in range(i, i + len(target_word_ids)):
                        label[j] = 1
                    break                

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        labels.append(label)
    
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids, input_mask, labels = tuple(t.to(device) for t in batch)
        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=input_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids, input_mask, labels = tuple(t.to(device) for t in batch)

            outputs = model(input_ids, attention_mask=input_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, f1