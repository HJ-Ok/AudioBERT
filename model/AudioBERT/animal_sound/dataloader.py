from functools import partial

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def find_span_position(span, offsets, sentence, tmp_index):
    try:
        start_idx = sentence.index(span)
        end_idx = start_idx + len(span)

        for i, (token_start, token_end) in enumerate(offsets):
            if token_start >= start_idx and token_end <= end_idx:
                return i
    except:
        if tmp_index == 0:
            return 0
        else:
            return len(offsets) - 1


def find_span_token_position(sentence, tokenized_sentence, offsets, span):
    span_token_pos = find_span_position(span, offsets, sentence, 0)

    span_first_token = tokenized_sentence[span_token_pos] if span_token_pos is not None else None

    return {"span_token_pos": span_token_pos, "span_first_token": span_first_token}


class QA_dataset(Dataset):
    def __init__(self, text, audio_path, animal, description, tokenizer):
        self.text = text
        self.audio_path = audio_path
        self.animal = animal
        self.description = description
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item]).lower()
        text = text.replace("[mask]", "[MASK]")
        audio_path = str(self.audio_path[item])
        audio_array, _ = librosa.load(audio_path, sr=16000)

        animal = self.animal[item]

        target = self.tokenizer.encode(animal, add_special_tokens=False)

        if len(target) > 1:
            mask_token = self.tokenizer.mask_token
            mask_str = " ".join([mask_token] * len(target))
            text_v2 = text.replace("[MASK]", mask_str)
        else:
            text_v2 = text

        query = self.tokenizer.cls_token + text_v2 + self.tokenizer.sep_token
        query_tokens = self.tokenizer.encode(query, add_special_tokens=False, return_offsets_mapping=True)
        tokenized_sentence = self.tokenizer.tokenize(query)

        input_ids = query_tokens["input_ids"]
        attention_mask = [1.0] * len(input_ids)
        offsets = query_tokens["offset_mapping"]

        results = find_span_token_positions(query, tokenized_sentence, offsets, str(self.description[item]).lower())
        if results["span_token_pos"] is None:
            span_token_pos = 0
        else:
            span_token_pos = results["span_token_pos"]

        target_text = text.replace("[MASK]", animal)

        target_query = self.tokenizer.cls_token + target_text + self.tokenizer.sep_token
        label = self.tokenizer.encode(target_query, add_special_tokens=False)

        if len(input_ids) > 512:
            label = label[:512]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
            "audio": audio_array,
            "target": target,
            "span_token_pos": span_token_pos,
        }


def dynamic_padding_collate_fn(batch, extractor):
    max_seq_len = max([len(item["input_ids"]) for item in batch])
    max_target_len = 10
    input_ids, attention_mask, labels, audios, targets, span_token_pos = [], [], [], [], [], []

    for item in batch:
        padded_input_ids = item["input_ids"] + [0] * (max_seq_len - len(item["input_ids"]))
        padded_attention_mask = item["attention_mask"] + [0.0] * (max_seq_len - len(item["attention_mask"]))
        input_ids.append(padded_input_ids)
        attention_mask.append(padded_attention_mask)
        padded_label_ids = item["label"] + [0] * (max_seq_len - len(item["label"]))
        labels.append(padded_label_ids)
        audios.append(item["audio"])
        padded_target_ids = item["target"] + [0] * (max_target_len - len(item["target"]))
        targets.append(padded_target_ids)
        span_token_pos.append(item["span_token_pos"])

    audio_feature = extractor(audios, sampling_rate=16000, return_tensors="pt")

    return {
        "encoder_input_ids": audio_feature.input_values,
        "span_token_pos": torch.tensor(span_token_pos, dtype=torch.long),
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
    }


def create_data_loader(df, tokenizer, extractor, batch_size, shuffle_=False, test_wiki=False):
    if test_wiki:
        ds = QA_dataset(
            text=df.sentence.to_numpy(),
            audio_path=df.audio_path.to_numpy(),
            animal=df.animal.to_numpy(),
            description=df.description.to_numpy(),
            tokenizer=tokenizer,
        )
    else:
        ds = QA_dataset(
            text=df.sentence.to_numpy(),
            audio_path=df.top_1_audio_path.to_numpy(),
            animal=df.animal.to_numpy(),
            description=df.description.to_numpy(),
            tokenizer=tokenizer,
        )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle=shuffle_,
        collate_fn=partial(dynamic_padding_collate_fn, extractor=extractor),
    )
