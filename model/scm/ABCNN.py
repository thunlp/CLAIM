import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from model.encoder.BertEncoder import BertEncoder
from model.layer.ABCNN import Abcnn1, Abcnn2, Abcnn3
from model.loss import cross_entropy_loss
from tools.accuracy_tool import single_label_top1_accuracy


class SCMABCNN(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(SCMABCNN, self).__init__()

        self.abcnn = Abcnn2(
            config.getint("model", "hidden_size"),
            config.getint("data", "max_seq_length "),
            3
        )

        self.criterion = cross_entropy_loss
        self.accuracy_function = single_label_top1_accuracy
        self.embedding = nn.Embedding(len(json.load(open(config.get("data", "word2id")))),
                                      config.getint("model", "hidden_size"))

    def init_multi_gpu(self, device, config, *args, **params):
        pass

    def forward(self, data, config, gpu_list, acc_result, mode):
        A = data['A']
        B = data['B']
        C = data['C']
        label = data["label"]

        A = self.embedding(A)
        B = self.embedding(B)
        C = self.embedding(C)
        batch = A.size()[0]
        l = A.size()[1]
        hidden = A.size()[2]
        A = A.view(batch, 1, l, hidden)
        B = B.view(batch, 1, l, hidden)
        C = C.view(batch, 1, l, hidden)

        b_s = self.abcnn(A, B)
        c_s = self.abcnn(A, C)
        s = torch.cat([b_s, c_s], dim=1)

        loss = self.criterion(s, label)
        acc_result = self.accuracy_function(s, label, config, acc_result)

        return {"loss": loss, "acc_result": acc_result}
