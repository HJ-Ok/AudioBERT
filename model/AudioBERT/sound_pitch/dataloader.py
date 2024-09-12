from functools import partial

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy as np
import librosa


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


def find_span_token_positions(sentence, tokenized_sentence, offsets, span1, span2):
    span1_token_pos = find_span_position(span1, offsets, sentence, 0)
    span2_token_pos = find_span_position(span2, offsets, sentence, 1)

    span1_first_token = tokenized_sentence[span1_token_pos] if span1_token_pos is not None else None
    span2_first_token = tokenized_sentence[span2_token_pos] if span2_token_pos is not None else None

    return {
        "span1_token_pos": span1_token_pos,
        "span1_first_token": span1_first_token,
        "span2_token_pos": span2_token_pos,
        "span2_first_token": span2_first_token,
    }


class QA_dataset(Dataset):
    def __init__(self, text, answer, span1, span2, id_1_audio_path, id_2_audio_path, tokenizer):
        self.text = text
        self.answer = answer
        self.span1 = span1
        self.span2 = span2
        self.id_1_audio_path = id_1_audio_path
        self.id_2_audio_path = id_2_audio_path
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item]).lower()
        text = text.replace("[mask]", "[MASK]")
        audio_path_1 = str(self.id_1_audio_path[item])
        audio_array_1, _ = librosa.load(audio_path_1, sr=16000)

        if len(audio_array_1) < 500:
            audio_array_1 = np.pad(audio_array_1, (0, 500 - len(audio_array_1)), mode="constant")

        audio_path_2 = str(self.id_2_audio_path[item])
        audio_array_2, _ = librosa.load(audio_path_2, sr=16000)

        if len(audio_array_2) < 500:
            audio_array_2 = np.pad(audio_array_2, (0, 500 - len(audio_array_2)), mode="constant")

        answer = self.answer[item]
        query = self.tokenizer.cls_token + text + self.tokenizer.sep_token
        query_tokens = self.tokenizer(query, add_special_tokens=False, return_offsets_mapping=True)
        tokenized_sentence = self.tokenizer.tokenize(query)

        input_ids = query_tokens["input_ids"]
        attention_mask = [1.0] * len(input_ids)
        offsets = query_tokens["offset_mapping"]

        results = find_span_token_positions(
            query, tokenized_sentence, offsets, str(self.span1[item]).lower(), str(self.span2[item]).lower()
        )
        if results["span1_token_pos"] is None:
            span1_token_pos = 0
        else:
            span1_token_pos = results["span1_token_pos"]

        if results["span2_token_pos"] is None:
            span2_token_pos = 0
        else:
            span2_token_pos = results["span2_token_pos"]

        first_token_pos = [span1_token_pos, span2_token_pos]
        # first_token_pos = [0, len(input_ids) - 1]

        target_text = text.replace("[MASK]", answer)
        target_query = self.tokenizer.cls_token + target_text + self.tokenizer.sep_token
        label = self.tokenizer.encode(target_query, add_special_tokens=False)

        target = self.tokenizer.encode(answer, add_special_tokens=False)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
            "first_token_pos": first_token_pos,
            "audio_1": audio_array_1,
            "audio_2": audio_array_2,
            "target": target,
        }


def dynamic_padding_collate_fn(batch, extractor):
    max_seq_len = max([len(item["input_ids"]) for item in batch])
    input_ids, attention_mask, labels, first_token_pos, audios_1, audios_2, targets = [], [], [], [], [], [], []

    for item in batch:
        padded_input_ids = item["input_ids"] + [0] * (max_seq_len - len(item["input_ids"]))
        padded_attention_mask = item["attention_mask"] + [0.0] * (max_seq_len - len(item["attention_mask"]))
        input_ids.append(padded_input_ids)
        attention_mask.append(padded_attention_mask)
        padded_label_ids = item["label"] + [0] * (max_seq_len - len(item["label"]))
        labels.append(padded_label_ids)
        first_token_pos.append(item["first_token_pos"])
        audios_1.append(item["audio_1"])
        audios_2.append(item["audio_2"])
        targets.append(item["target"])
    audio_feature_1 = extractor(audios_1, sampling_rate=16000, return_tensors="pt", padding=True)
    audio_feature_2 = extractor(audios_2, sampling_rate=16000, return_tensors="pt", padding=True)

    return {
        "encoder_input_ids_1": audio_feature_1.input_values,
        "encoder_input_ids_2": audio_feature_2.input_values,
        "first_token_pos": torch.tensor(first_token_pos, dtype=torch.long),
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
    }


def create_data_loader(df, tokenizer, extractor, batch_size, shuffle_=False):
    ds = QA_dataset(
        text=df.sentence.to_numpy(),
        answer=df.answer.to_numpy(),
        span1=df.span1.to_numpy(),
        span2=df.span2.to_numpy(),
        id_1_audio_path=df.id_1_audio_path.to_numpy(),
        id_2_audio_path=df.id_2_audio_path.to_numpy(),
        tokenizer=tokenizer,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle=shuffle_,
        collate_fn=partial(dynamic_padding_collate_fn, extractor=extractor),
    )
