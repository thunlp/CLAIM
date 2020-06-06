import json
import torch
import os
import numpy as np

from pytorch_pretrained_bert.tokenization import BertTokenizer

from formatter.Basic import BasicFormatter


class BertSCM(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = BertTokenizer.from_pretrained(config.get("model", "bert_path"))
        self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode

    def process(self, data, config, mode, *args, **params):
        A = []
        B = []
        C = []
        label = []

        for temp in data:
            for name in ["A", "B", "C"]:
                text = temp[name]

                text = self.tokenizer.tokenize(text)
                while len(text) < self.max_len:
                    text.append("[PAD]")
                text = text[0:self.max_len]
                if name == "A":
                    A.append(self.tokenizer.convert_tokens_to_ids(text))
                elif name == "B":
                    B.append(self.tokenizer.convert_tokens_to_ids(text))
                else:
                    C.append(self.tokenizer.convert_tokens_to_ids(text))

            if temp["label"] == "B":
                label.append(0)
            else:
                label.append(1)

        A = torch.LongTensor(A)
        B = torch.LongTensor(B)
        C = torch.LongTensor(C)
        label = torch.LongTensor(label)

        return {"A": A, "B": B, "C": C, "label": label}
