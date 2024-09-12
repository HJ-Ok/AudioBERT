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

        encoder_input_ids = d["encoder_input_ids"].to(device)
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        span_token_pos = d["span_token_pos"].to(device)
        label = d["labels"].to(device)
        targets = d["targets"].to(device)

        outputs, predicts = model(
            audio_features=encoder_input_ids, input_ids=input_ids, span_token_pos=span_token_pos, targets=label
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
    outputs_arr = []
    cnt = 0
    for d in data_loader:
        with torch.no_grad():
            encoder_input_ids = d["encoder_input_ids"].to(device)
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            span_token_pos = d["span_token_pos"].to(device)
            label = d["labels"].to(device)
            targets = d["targets"].to(device)

            outputs, predicts = model(
                audio_features=encoder_input_ids, input_ids=input_ids, span_token_pos=span_token_pos, targets=label
            )
            losses.append(outputs.loss.item())
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if cnt == 0:
                cnt += 1
                labels_arr = targets
            else:
                labels_arr = torch.cat([labels_arr, targets], 0)
            for i in range(len(predicts)):
                outputs_arr.append(predicts[i])

    f1_acc = calc_f1_acc(outputs_arr, labels_arr)
    return outputs_arr, np.mean(losses), f1_acc
