import numpy as np
import torch
import torch.nn as nn
from utils import AverageMeter, calc_f1_acc


def train_epoch(model, data_loader, optimizer, device, scheduler, epoch):
    losses = AverageMeter()
    model = model.train()

    correct_predictions = 0

    for step, d in enumerate(data_loader):
        batch_size = d["input_ids"].size(0)

        encoder_input_ids_1 = d["encoder_input_ids_1"].to(device)
        encoder_input_ids_2 = d["encoder_input_ids_2"].to(device)
        first_token_pos = d["first_token_pos"].to(device)
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        label = d["labels"].to(device)
        targets = d["targets"].to(device)

        outputs, predicts = model(
            audio_features_1=encoder_input_ids_1,
            audio_features_2=encoder_input_ids_2,
            first_token_pos=first_token_pos,
            input_ids=input_ids,
            targets=label,
        )

        loss = outputs.loss
        correct_predictions += calc_f1_acc(predicts, targets)
        losses.update(loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    return correct_predictions / (step + 1), losses.avg


def validate(model, data_loader, device):
    model = model.eval()
    losses = []
    cnt = 0
    for d in data_loader:
        with torch.no_grad():
            encoder_input_ids_1 = d["encoder_input_ids_1"].to(device)
            encoder_input_ids_2 = d["encoder_input_ids_2"].to(device)
            first_token_pos = d["first_token_pos"].to(device)
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            label = d["labels"].to(device)
            targets = d["targets"].to(device)

            outputs, predicts = model(
                audio_features_1=encoder_input_ids_1,
                audio_features_2=encoder_input_ids_2,
                first_token_pos=first_token_pos,
                input_ids=input_ids,
                targets=label,
            )
            losses.append(outputs.loss.item())
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if cnt == 0:
                cnt += 1
                outputs_arr = predicts
                labels_arr = targets
            else:
                outputs_arr = torch.cat([outputs_arr, predicts], 0)
                labels_arr = torch.cat([labels_arr, targets], 0)

    f1_acc = calc_f1_acc(outputs_arr, labels_arr)
    return outputs_arr, np.mean(losses), f1_acc
