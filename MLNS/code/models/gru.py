# -*- coding: utf-8 -*-

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch import nn

class GRU_Model(nn.Module):
    def __init__(self, config):
        super(GRU_Model, self).__init__()
        self.config = config
        self.gru = nn.GRU(self.config.feature_dim, self.config.hidden_size, batch_first = True)
        self.linear = nn.Linear(self.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.dropout_embed = nn.Dropout(0.2)
        self.dropout_rep = nn.Dropout(0.8)

        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.linear.bias, 0.1)

    def forward(self, x_train):
        x_train = x_train.cuda()
        x_train = x_train.view(-1, self.config.seq_len, self.config.feature_dim)
        # x_train = self.dropout_embed(x_train)
        gru_out, ht = self.gru(x_train)
        ht = ht.view(-1, self.config.hidden_size)
        ht = self.dropout_rep(ht)
        out = self.sigmoid(self.linear(ht))
        return out.cpu()




