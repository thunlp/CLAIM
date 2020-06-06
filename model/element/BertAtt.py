import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert import BertModel
from model.layer.Attention import Attention
from model.loss import MultiLabelSoftmaxLoss
from tools.accuracy_tool import multi_label_accuracy


class BertAtt(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BertAtt, self).__init__()

        self.hidden_size = config.getint("model", "hidden_size")

        self.encoder = BertModel.from_pretrained(config.get("model", "bert_path"))
        self.attention = Attention(config, gpu_list, *args, **params)

        self.rank_module = nn.Linear(self.hidden_size * 2 * 20, 20 * 2)

        self.criterion = MultiLabelSoftmaxLoss(config, 20)
        self.accuracy_function = multi_label_accuracy

    def init_multi_gpu(self, device, config, *args, **params):
        self.bert = nn.DataParallel(self.bert, device_ids=device)
        self.fc = nn.DataParallel(self.fc, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        context = data["text"]
        tags = data["tags"]

        batch = tags.size()[0]
        num_tags = tags.size()[1]

        context = context.view(batch, -1)
        tags = tags[0].view(20, -1)

        context, _ = self.encoder(context)
        tags, _ = self.encoder(tags)
        context = context[-1]
        tags = tags[-1]

        context = context.view(batch, 1, -1, self.hidden_size).repeat(1, num_tags, 1, 1)
        tags = tags.view(1, 20, -1, self.hidden_size).repeat(batch, 1, 1, 1)

        context = context.view(batch * 20, -1, self.hidden_size)
        tags = tags.view(batch * 20, -1, self.hidden_size)

        c, q, a = self.attention(context, tags)

        y = torch.cat([torch.max(c, dim=1)[0], torch.max(q, dim=1)[0]], dim=1)

        y = y.view(batch, -1)
        y = self.rank_module(y)

        y = y.view(batch, 20, 2)

        loss = self.criterion(y, data["label"])
        acc_result = self.accuracy_function(y, data["label"], config, acc_result)

        return {"loss": loss, "acc_result": acc_result}
