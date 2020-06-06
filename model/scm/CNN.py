import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from model.encoder.CNNEncoder import CNNEncoder
from model.loss import cross_entropy_loss
from tools.accuracy_tool import single_label_top1_accuracy


class SCMCNN(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(SCMCNN, self).__init__()

        self.encoder = CNNEncoder(config, gpu_list, *args, **params)
        self.fc = nn.Bilinear(config.getint("model", "hidden_size"), config.getint("model", "hidden_size"), 1)

        self.criterion = cross_entropy_loss
        self.accuracy_function = single_label_top1_accuracy
        self.embedding = nn.Embedding(len(json.load(open(config.get("data", "word2id")))),
                                      config.getint("model", "hidden_size"))

    def init_multi_gpu(self, device, config, *args, **params):
        pass
        # self.bert = nn.DataParallel(self.bert, device_ids=device)
        # self.fc = nn.DataParallel(self.fc, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        A = data['A']
        B = data['B']
        C = data['C']
        label = data["label"]
        A = self.embedding(A)
        B = self.embedding(B)
        C = self.embedding(C)
        A = self.encoder(A)
        B = self.encoder(B)
        C = self.encoder(C)

        b_s = self.fc(A, B)
        c_s = self.fc(A, C)
        s = torch.cat([b_s, c_s], dim=1)

        loss = self.criterion(s, label)
        acc_result = self.accuracy_function(s, label, config, acc_result)

        return {"loss": loss, "acc_result": acc_result}
