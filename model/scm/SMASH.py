import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from model.encoder.LSTMEncoder import LSTMEncoder
from model.layer.Attention import Attention
from model.loss import cross_entropy_loss
from tools.accuracy_tool import single_label_top1_accuracy


class DEC(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(DEC, self).__init__()

        self.sentence_encoder = LSTMEncoder(config, gpu_list, *args, **params)
        self.document_encoder = LSTMEncoder(config, gpu_list, *args, **params)
        self.sentence_attention = Attention(config, gpu_list, *args, **params)
        self.document_attention = Attention(config, gpu_list, *args, **params)

    def forward(self, x):
        batch = x.size()[0]
        sent = x.size()[1]
        word = x.size()[2]
        hidden = x.size()[3]

        _, h1 = self.sentence_encoder(x.view(batch * sent, word, hidden))

        x, y, a = self.sentence_attention(h1, h1)

        x = x.view(batch, sent, word, hidden)
        x = torch.max(x, dim=2)[0]

        _, h2 = self.document_encoder(x)

        _, y, a = self.document_attention(h2, h2)
        y = torch.max(h2, dim=1)[0]

        return torch.cat([torch.max(x, dim=1)[0], y], dim=1)


class SCMSMASH(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(SCMSMASH, self).__init__()

        self.encoder = DEC(config, gpu_list, *args, **params)
        self.fc = nn.Linear(config.getint("model", "hidden_size") * 4, 1)

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

        b_s = self.fc(torch.cat([A, B], dim=1))
        c_s = self.fc(torch.cat([A, C], dim=1))
        s = torch.cat([b_s, c_s], dim=1)

        loss = self.criterion(s, label)
        acc_result = self.accuracy_function(s, label, config, acc_result)

        return {"loss": loss, "acc_result": acc_result}
