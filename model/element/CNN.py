import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from model.encoder.CNNEncoder import CNNEncoder
from model.loss import MultiLabelSoftmaxLoss
from tools.accuracy_tool import multi_label_accuracy


class CNN(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(CNN, self).__init__()

        self.encoder = CNNEncoder(config, gpu_list, *args, **params)
        self.hidden_size = config.getint("model", "hidden_size")
        self.word_num = 0
        f = open(config.get("data", "word2id"), "r", encoding="utf8")
        for line in f:
            self.word_num += 1
        self.embedding = nn.Embedding(self.word_num, self.hidden_size)
        self.fc = nn.Linear(768, 20 * 2)

        self.criterion = MultiLabelSoftmaxLoss(config, 20)
        self.accuracy_function = multi_label_accuracy

    def init_multi_gpu(self, device, config, *args, **params):
        return

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['text']
        x = self.embedding(x)
        y = self.encoder(x)
        result = self.fc(y)
        result = result.view(result.size()[0], -1, 2)

        loss = self.criterion(result, data["label"])
        acc_result = self.accuracy_function(result, data["label"], config, acc_result)

        return {"loss": loss, "acc_result": acc_result}
