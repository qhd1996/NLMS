# -*- coding: utf-8 -*-
import gc
import time

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from code.util.config import Config
from code.models.gru import GRU_Model
from code.util.util import custom_loss, EarlyStopping

from code.util.util import load_data

print(os.listdir("../../input"))

data, label = load_data("../../input/train_data.npy")
count = 0
for i in label:
    if i == 1:
        count += 1
print("Normal", label.shape[0] - count, "Intrusion", count)
data = data[:, :, np.newaxis]
label = label[:, np.newaxis]
print(label)
# split
indices = np.zeros(data.shape[0], dtype=int)
for i in range(indices.shape[0]):
    indices[i] = i
np.random.shuffle(indices)
tr_ind, test_ind, label_train, y_test = train_test_split(indices, label[:, 0], stratify = label[:, 0], test_size = 0.2, random_state = 42)
tr_ind, val_ind, label_train, label_val = train_test_split(tr_ind, label_train, stratify = label_train, test_size = 0.125, random_state = 42)
train_x = data[tr_ind, : , :]
train_label = label[tr_ind, :]
valid_x = data[val_ind, : ]
valid_label = label[val_ind, : ]
test_x = data[val_ind, :, :]
test_label = label[val_ind, :]
print("Train.shape", train_x.shape, "Train label.shape", train_label.shape)
print("Valid.shape", valid_x.shape, "Valid label.shape", valid_label.shape)
print("Test.shape", test_x.shape, "Test label.shape", test_label.shape)

def train_model(model, train_x, train_label, valid_x, valid_label, test_x, test_label,loss_fn):
    train_x = torch.from_numpy(train_x).float()
    train_label = torch.from_numpy(train_label).float()
    valid_x = torch.from_numpy(valid_x).float()
    valid_label = torch.from_numpy(valid_label).float()
    test_x = torch.from_numpy(test_x).float()
    test_label = torch.from_numpy(test_label).float()
    train_dataset = torch.utils.data.TensorDataset(train_x, train_label)
    valid_dataset = torch.utils.data.TensorDataset(valid_x, valid_label)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=model.config.lr)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    valid_losses = []
    avg_valid_losses = []
    for epoch in range(model.config.num_epochs):
        print("EPOCH {} starts!".format(epoch))
        start = time.time()
        train_loader = DataLoader(dataset = train_dataset, batch_size = model.config.batch_size, shuffle = True)
        valid_loader = DataLoader(dataset = valid_dataset, batch_size = model.config.batch_size, shuffle = False)
        model.train()
        avg_loss = 0
        # run batches
        count = 0
        for i, data in enumerate(train_loader):
            x_train, label_train = data
            out = model(x_train)
            loss = loss_fn(out, label_train)
            # print(i + 1, 'loss: ',loss.item())
            avg_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1
        print('average loss:', avg_loss / count)
        gc.collect()
        with torch.no_grad():
            model.eval()
            preds_valid = []
            for i, data in enumerate(valid_loader):
                x_valid, label_valid = data
                out = model(x_valid)
                preds_valid.append(out)
                loss = loss_fn(out, label_valid)
                # print(i + 1, 'loss: ',loss.item())
                valid_losses.append(loss.item())

            valid_loss = np.average(valid_losses)
            avg_valid_losses.append(valid_loss)
            valid_losses = []
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    with torch.no_grad():
        torch.cuda.empty_cache()
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load('checkpoint.pt'))
        model = model.cuda()
        model.eval()
        N = model.config.batch_size
        preds_test = np.zeros((test_label.shape[0], test_label.shape[1]))
        batches_test = int(test_label.shape[0] // N)
        for i in range(batches_test):
            batch_start = (i * N)
            batch_end = (i + 1) * N
            x_batch = test_x[batch_start:batch_end, :, :]
            out = model(x_batch)
            preds_test[batch_start:batch_end, :] = out.detach().numpy()
        if (batch_end < test_x.shape[0]):
            x_batch = test_x[batch_end:, :, :]
            out = model(x_batch)
            preds_test[batch_end:, :] = out.detach().numpy()
    preds_test  = preds_test >= 0.5
    accuracy = accuracy_score(test_label.long(), preds_test)
    precision = precision_score(test_label.long(), preds_test, average = "macro")
    recall = recall_score(test_label.long(), preds_test, average = "macro")
    f1 = f1_score(test_label.long(), preds_test, average = "macro")
    end = time.time()
    print('"accuracy: " {}\t " precision: "{}\t "recall: " {}\t "f1_score: "{}\t time={:.2f}s'.format(accuracy, precision, recall, f1, end - start))

def seed_everything(seed=0):
    torch.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    torch.manual_seed(0)
    config = Config()
    model = GRU_Model(config)
    model = model.cuda()
    train_model(model, train_x, train_label, valid_x, valid_label, test_x, test_label, custom_loss)
