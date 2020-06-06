import json
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter


class SentSCM(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = json.load(open(config.get("data", "word2id"), "r", encoding="utf8"))
        self.max_len = config.getint("data", "sent_len")
        self.max_sent = config.getint("data", "max_sent")
        self.mode = mode

    def convert_tokens_to_ids(self, text):
        arr = [[]]
        for a in range(0, len(text)):
            if text[a] in ["，", "。"]:
                arr.append([])
                continue
            if text[a] in self.tokenizer.keys():
                arr[-1].append(self.tokenizer[text[a]])
            else:
                arr[-1].append(self.tokenizer["[UNK]"])
        while len(arr) < self.max_sent:
            arr.append([])
        arr = arr[:self.max_sent]
        for a in range(0, len(arr)):
            while len(arr[a]) < self.max_len:
                arr[a].append(self.tokenizer["[PAD]"])
            arr[a] = arr[a][:self.max_len]
        return arr

    def process(self, data, config, mode, *args, **params):
        A = []
        B = []
        C = []
        label = []

        for temp in data:
            for name in ["A", "B", "C"]:
                text = temp[name]

                text = text[0:self.max_len]
                if name == "A":
                    A.append(self.convert_tokens_to_ids(text))
                elif name == "B":
                    B.append(self.convert_tokens_to_ids(text))
                else:
                    C.append(self.convert_tokens_to_ids(text))

            if temp["label"] == "B":
                label.append(0)
            else:
                label.append(1)

        A = torch.LongTensor(A)
        B = torch.LongTensor(B)
        C = torch.LongTensor(C)
        label = torch.LongTensor(label)

        return {"A": A, "B": B, "C": C, "label": label}
