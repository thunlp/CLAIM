import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder.BertEncoder import BertEncoder
from model.loss import MultiLabelSoftmaxLoss
from tools.accuracy_tool import multi_label_accuracy


class Bert(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Bert, self).__init__()

        self.bert = BertEncoder(config, gpu_list, *args, **params)
        self.fc = nn.Linear(768, 20 * 2)

        self.criterion = MultiLabelSoftmaxLoss(config, 20)
        self.accuracy_function = multi_label_accuracy

    def init_multi_gpu(self, device, config, *args, **params):
        self.bert = nn.DataParallel(self.bert, device_ids=device)
        self.fc = nn.DataParallel(self.fc, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['text']
        y = self.bert(x)
        result = self.fc(y)
        result = result.view(result.size()[0], -1, 2)

        loss = self.criterion(result, data["label"])
        acc_result = self.accuracy_function(result, data["label"], config, acc_result)

        return {"loss": loss, "acc_result": acc_result}
