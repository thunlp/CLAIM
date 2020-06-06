import torch
import torch.nn as nn
import json
import os

from model.loss import MultiLabelSoftmaxLoss
from tools.accuracy_tool import multi_label_accuracy


class ResnetBlock(nn.Module):
    def __init__(self, channel_size):
        super(ResnetBlock, self).__init__()
        self.channel_size = channel_size
        self.maxpool = nn.Sequential(
            nn.ConstantPad1d(padding=(0, 1), value=0),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,
                      kernel_size=3, padding=1),
        )

    def forward(self, x):
        x_shortcut = self.maxpool(x)
        x = self.conv(x_shortcut)
        x = x + x_shortcut
        return x


class DPCNN(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(DPCNN, self).__init__()
        self.model_name = "DPCNN"
        self.emb_dim = config.getint("model", "hidden_size")
        self.mem_dim = config.getint("model", "hidden_size")
        self.output_dim = 40
        self.word_num = 0
        f = open(config.get("data", "word2id"), "r", encoding="utf8")
        for line in f:
            self.word_num += 1

        self.embedding = nn.Embedding(self.word_num, self.emb_dim)

        # region embedding
        self.region_embedding = nn.Sequential(
            nn.Conv1d(self.emb_dim, self.mem_dim,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.mem_dim),
            nn.ReLU(),
        )
        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(num_features=self.mem_dim),
            nn.ReLU(),
            nn.Conv1d(self.mem_dim, self.mem_dim,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.mem_dim),
            nn.ReLU(),
            nn.Conv1d(self.mem_dim, self.mem_dim,
                      kernel_size=3, padding=1),
        )

        self.num_seq = config.getint("data", "max_seq_length")
        resnet_block_list = []
        while (self.num_seq > 2):
            resnet_block_list.append(ResnetBlock(self.mem_dim))
            self.num_seq = self.num_seq // 2
        self.resnet_layer = nn.Sequential(*resnet_block_list)
        self.fc = nn.Sequential(
            nn.Linear(self.mem_dim * self.num_seq, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_dim, self.output_dim)
        )

        self.criterion = MultiLabelSoftmaxLoss(config, 20)
        self.accuracy_function = multi_label_accuracy

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['text']

        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.region_embedding(x)
        x = self.conv_block(x)
        x = self.resnet_layer(x)
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(x.size()[0], -1)
        out = self.fc(x)
        result = out.view(out.size()[0], -1, 2)

        loss = self.criterion(result, data["label"])
        acc_result = self.accuracy_function(result, data["label"], config, acc_result)

        return {"loss": loss, "acc_result": acc_result}
