import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch.surrogate as surrogate
import snntorch as snn
import numpy as np
from utils.config import INITIAL_MEAN_DICT
from utils.monitor import Monitor

class TextCNNEmbeded(nn.Module):
    def __init__(self, args, spike_grad=surrogate.fast_sigmoid(slope=25)) -> None:
        super().__init__()
        self.dead_neuron_checker = args.dead_neuron_checker
        self.initial_method = args.initial_method
        self.positive_init_rate = args.positive_init_rate
        self.hidden_dim = args.hidden_dim
        # random embedding
        self.embedding = nn.Embedding(len(args.word2id), args.hidden_dim)
        self.convs_1 = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=args.filter_num, kernel_size=(filter_size, args.hidden_dim))
            for filter_size in args.filters
        ])
        self.middle_relu = nn.ModuleList([
            nn.ReLU()
            for _ in args.filters
        ])
        # self.maxpool_1 = nn.ModuleList([
        #     nn.MaxPool2d((args.sentence_length - filter_size + 1, 1)) for filter_size in args.filters
        # ])
        self.avgpool_1 = nn.ModuleList([
            nn.AvgPool2d((args.sentence_length - filter_size + 1, 1)) for filter_size in args.filters
        ])
        self.relu_2 = nn.ReLU()
        self.drop = nn.Dropout(p=args.dropout_p)
        self.fc_1 = nn.Linear(len(args.filters)*args.filter_num, args.label_num, bias=False)
    
    def initial(self):
        for c in self.convs_1:
            c.weight.data.add_(INITIAL_MEAN_DICT['conv-kaiming'][self.positive_init_rate])
        m = self.fc_1
        m.weight.data.add_(INITIAL_MEAN_DICT["linear-kaiming"][self.positive_init_rate])

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(dim=1)
        x = self.embedding(x)

        mean_value = torch.mean(x)
        variance_value = torch.var(x)
        x = torch.clip((x - mean_value) / 6 / torch.sqrt(variance_value) + 0.5, 0, 1)
        # print(x)
        x = x.float()
        conv_out = [conv(x) for conv in self.convs_1]
        conv_out = [self.middle_relu[i](conv_out[i]) for i in range(len(self.middle_relu))]
        # pooled_out = [self.maxpool_1[i](conv_out[i]) for i in range(len(self.maxpool_1))]
        pooled_out = [self.avgpool_1[i](conv_out[i]) for i in range(len(self.avgpool_1))]
        pooled_out = [self.relu_2(pool) for pool in pooled_out]
        flatten = torch.cat(pooled_out, dim=1).view(batch_size, -1)
        flatten = self.drop(flatten)
        fc_output = self.fc_1(flatten)
        return fc_output
        
