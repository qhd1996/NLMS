# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn

def load_data(dataset):

    data = np.load(dataset)

    # get the labels from the dataset
    labels = data[:, 17]
    labels = labels.astype(np.float32)

    # get the features from the dataset
    data = np.delete(arr=data, obj=[17], axis=1)
    data = data.astype(np.float32)

    return data, labels

if __name__ == "__main__":
    data, label = load_data("../input/train_data.npy")
    print(data.shape)
    print(label.shape)

def custom_loss(data, targets):
    loss = nn.BCELoss()(data, targets)
    return loss

def hinge_loss(model, data, targets):
    one_hot = -1 * torch.ones(data.shape[0], data.shape[1])
    one_hot = one_hot.scatter_(1, targets.long(), 1)
    error = torch.max(torch.zeros(data.shape[0], data.shape[1]), 1 - one_hot.float() * data.float())
    error = torch.mul(error, error)
    weight = model.svm.linear.weight.cpu()
    loss =  (1 / 2 * torch.mul(weight, weight)).sum() + model.config.svm_c * error.sum()
    loss = loss
    # loss = model.config.svm_c * error.sum()
    return loss

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        state_dict = model.state_dict()
        torch.save(state_dict, './checkpoint.pt')
        self.val_loss_min = val_loss
