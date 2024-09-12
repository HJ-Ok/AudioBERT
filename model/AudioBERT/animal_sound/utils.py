import math
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


def calc_f1_acc(pred, label):
    x = label.cpu().numpy()

    correct_predictions = []

    for i in range(len(x)):
        actual_tokens = x[i][x[i] != 0]
        predicted_tokens = pred[i].cpu().numpy()

        if len(actual_tokens) != len(predicted_tokens):
            print(actual_tokens)
            print(predicted_tokens)
        if len(actual_tokens) == len(predicted_tokens) and all(actual_tokens == predicted_tokens):
            correct_predictions.append(1)
        else:
            correct_predictions.append(0)
    acc = sum(correct_predictions) / len(correct_predictions)

    return acc


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))
