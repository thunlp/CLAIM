import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel

from tools.accuracy_tool import single_label_top1_accuracy


class BertQA(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BertQA, self).__init__()

        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))
        self.rank_module = nn.Linear(768 * config.getint("data", "topk"), 1)

        self.criterion = nn.CrossEntropyLoss()

        self.multi = config.getboolean("data", "multi_choice")
        self.multi_module = nn.Linear(4, 16)
        self.accuracy_function = single_label_top1_accuracy

    def init_multi_gpu(self, device, config, *args, **params):
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        text = data["text"]
        token = data["token"]
        mask = data["mask"]

        batch = text.size()[0]
        option = text.size()[1]
        k = config.getint("data", "topk")
        option = option // k
        text = text.view(text.size()[0] * text.size()[1], text.size()[2])
        token = token.view(token.size()[0] * token.size()[1], token.size()[2])
        mask = mask.view(mask.size()[0] * mask.size()[1], mask.size()[2])

        encode, y = self.bert.forward(text, token, mask, output_all_encoded_layers=False)

        y = y.view(batch * option, -1)
        y = self.rank_module(y)

        y = y.view(batch, option)

        if self.multi:
            y = self.multi_module(y)

        label = data["label"]
        loss = self.criterion(y, label)
        acc_result = self.accuracy_function(y, label, config, acc_result)
        return {"loss": loss, "acc_result": acc_result}
