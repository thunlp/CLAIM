import json
import torch
import os
import numpy as np

from pytorch_pretrained_bert.tokenization import BertTokenizer

from formatter.Basic import BasicFormatter


class BertEle(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = BertTokenizer.from_pretrained(config.get("model", "bert_path"))
        self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode

        self.tags = []
        self.tag = config.getboolean("data", "tags")
        if self.tag:
            path = config.get("data", "tag_path")
            f = open(path, "r", encoding="utf8")
            for line in f:
                self.tags.append(line[:-1])

            for a in range(0, len(self.tags)):
                self.tags[a] = self.tokenizer.tokenize(self.tags[a])
                while len(self.tags[a]) < 32:
                    self.tags[a].append("[PAD]")
                self.tags[a] = self.tags[a][:32]
                self.tags[a] = self.tokenizer.convert_tokens_to_ids(self.tags[a])

    def process(self, data, config, mode, *args, **params):
        input = []
        label = []
        tags = []

        for temp in data:
            text = temp["sentence"]

            text = self.tokenizer.tokenize(text)
            while len(text) < self.max_len:
                text.append("[PAD]")
            text = text[0:self.max_len]
            input.append(self.tokenizer.convert_tokens_to_ids(text))

            temp_label = np.zeros([20], dtype=np.long)
            for x in temp["labels"]:
                temp_label[int(x[2:]) - 1] = 1

            label.append(temp_label.tolist())

            if self.tag:
                tags.append(self.tags)

        input = torch.LongTensor(input)
        label = torch.LongTensor(label)
        tags = torch.LongTensor(tags)

        return {'text': input, 'label': label, 'tags': tags}
