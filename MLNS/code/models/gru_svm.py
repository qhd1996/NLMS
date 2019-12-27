# -*- coding: utf-8 -*-

import os

from code.models.svm import LinearSVM

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torch import nn

class GRU_SVM_Model(nn.Module):
    def __init__(self, config):
        super(GRU_SVM_Model, self).__init__()
        self.config = config
        self.gru = nn.GRU(self.config.feature_dim, self.config.hidden_size, batch_first = True)
        self.linear = nn.Linear(self.config.hidden_size, 1)
        self.svm = LinearSVM(self.config.hidden_size, self.config.num_classes)

        self.dropout_embed = nn.Dropout(0.2)
        self.dropout_rep = nn.Dropout(0.85)

        nn.init.normal_(self.svm.linear.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.svm.linear.bias, 0.1)

    def forward(self, x_train):
        x_train = x_train.cuda()
        x_train = x_train.view(-1, self.config.seq_len, self.config.feature_dim)
        gru_out, ht = self.gru(x_train)
        ht = ht.view(-1, self.config.hidden_size)
        ht = self.dropout_rep(ht)
        out = self.svm(ht)
        return out.cpu()




