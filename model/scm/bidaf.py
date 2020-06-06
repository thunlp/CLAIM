import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from model.encoder.LSTMEncoder import LSTMEncoder
from model.layer.Attention import Attention
from model.loss import cross_entropy_loss
from tools.accuracy_tool import single_label_top1_accuracy


class BiDAF(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BiDAF, self).__init__()

        self.hidden_size = config.getint("model", "hidden_size")

        self.embedding = nn.Embedding(len(json.load(open(config.get("data", "word2id")))),
                                      config.getint("model", "hidden_size"))
        self.encoder = LSTMEncoder(config, gpu_list, *args, **params)
        self.attention = Attention(config, gpu_list, *args, **params)

        self.fc = nn.Linear(self.hidden_size * 2, 1)

        self.criterion = cross_entropy_loss
        self.accuracy_function = single_label_top1_accuracy

    def init_multi_gpu(self, device, config, *args, **params):
        pass
        # self.bert = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        A = data["A"]
        B = data["B"]
        C = data["C"]
        label = data["label"]

        batch = A.size()[0]

        A = self.embedding(A)
        B = self.embedding(B)
        C = self.embedding(C)

        _, A = self.encoder(A)
        _, B = self.encoder(B)
        _, C = self.encoder(C)

        c1, q1, a1 = self.attention(A, B)
        y1 = torch.cat([torch.max(c1, dim=1)[0], torch.max(q1, dim=1)[0]], dim=1)
        y1 = y1.view(batch, -1)
        y1 = self.fc(y1)

        c2, q2, a2 = self.attention(A, C)
        y2 = torch.cat([torch.max(c2, dim=1)[0], torch.max(q2, dim=1)[0]], dim=1)
        y2 = y2.view(batch, -1)
        y2 = self.fc(y2)

        y = torch.cat([y1, y2], dim=1)

        loss = self.criterion(y, label)
        acc_result = self.accuracy_function(y, label, config, acc_result)

        return {"loss": loss, "acc_result": acc_result}
