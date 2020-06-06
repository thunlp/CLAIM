import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder.LSTMEncoder import LSTMEncoder
from model.layer.Attention import Attention
from model.loss import MultiLabelSoftmaxLoss
from tools.accuracy_tool import multi_label_accuracy


class BiDAF(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BiDAF, self).__init__()

        self.hidden_size = config.getint("model", "hidden_size")
        self.word_num = 0
        f = open(config.get("data", "word2id"), "r", encoding="utf8")
        for line in f:
            self.word_num += 1

        self.embedding = nn.Embedding(self.word_num, self.hidden_size)
        self.context_encoder = LSTMEncoder(config, gpu_list, *args, **params)
        self.question_encoder = LSTMEncoder(config, gpu_list, *args, **params)
        self.attention = Attention(config, gpu_list, *args, **params)

        self.rank_module = nn.Linear(self.hidden_size * 2 * 20, 20 * 2)

        self.criterion = MultiLabelSoftmaxLoss(config, 20)
        self.accuracy_function = multi_label_accuracy

    def init_multi_gpu(self, device, config, *args, **params):
        pass
        # self.bert = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        context = data["text"]
        tags = data["tags"]

        batch = tags.size()[0]
        num_tags = tags.size()[1]

        context = context.view(batch, 1, -1).repeat(1, num_tags, 1).view(batch * num_tags, -1)
        tags = tags.view(batch * num_tags, -1)
        context = self.embedding(context)
        tags = self.embedding(tags)

        _, context = self.context_encoder(context)
        _, tags = self.question_encoder(tags)

        c, q, a = self.attention(context, tags)

        y = torch.cat([torch.max(c, dim=1)[0], torch.max(q, dim=1)[0]], dim=1)

        y = y.view(batch, -1)
        y = self.rank_module(y)

        y = y.view(batch, 20, 2)

        loss = self.criterion(y, data["label"])
        acc_result = self.accuracy_function(y, data["label"], config, acc_result)

        return {"loss": loss, "acc_result": acc_result}
