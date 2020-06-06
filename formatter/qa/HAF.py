import json
import torch
import numpy as np
import os
from pytorch_pretrained_bert import BertTokenizer


class HAFQA:
    def __init__(self, config, mode):
        self.question_len = config.getint("data", "question_len")
        self.option_len = config.getint("data", "option_len")
        self.passage_len = config.getint("data", "passage_len")

        self.tokenizer = BertTokenizer.from_pretrained(config.get("data", "word2id"))
        self.k = config.getint("data", "topk")

    def convert(self, tokens, l):
        tokens = self.tokenizer.tokenize(tokens)
        while len(tokens) < l:
            tokens.append("[PAD]")
        tokens = tokens[:l]
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        return ids

    def process(self, data, config, mode, *args, **params):
        passage = []
        question = []
        option = []
        label = []

        for temp_data in data:
            if config.getboolean("data", "multi_choice"):
                label_x = 0
                if "A" in temp_data["answer"]:
                    label_x += 1
                if "B" in temp_data["answer"]:
                    label_x += 2
                if "C" in temp_data["answer"]:
                    label_x += 4
                if "D" in temp_data["answer"]:
                    label_x += 8
            else:
                label_x = 0
                if "A" in temp_data["answer"]:
                    label_x = 0
                if "B" in temp_data["answer"]:
                    label_x = 1
                if "C" in temp_data["answer"]:
                    label_x = 2
                if "D" in temp_data["answer"]:
                    label_x = 3

            label.append(label_x)

            temp_passage = []
            temp_option = []
            question.append(self.convert(temp_data["statement"], self.question_len))

            for option_ in ["A", "B", "C", "D"]:
                temp_option.append(self.convert(temp_data["option_list"][option_], self.option_len))

                ref = []
                k = [0, 1, 2, 6, 12, 7, 13, 3, 8, 9, 14, 15, 4, 10, 16, 5, 16, 17]
                for a in range(0, self.k):
                    res = temp_data["reference"][option_][k[a]]

                    ref.append(self.convert(res, self.passage_len))

                temp_passage.append(ref)

            passage.append(temp_passage)
            option.append(temp_option)

        question = torch.LongTensor(question)
        passage = torch.LongTensor(passage)
        option = torch.LongTensor(option)
        label = torch.LongTensor(np.array(label, dtype=np.int32))

        return {"passage": passage, "option": option, "question": question, 'label': label}
