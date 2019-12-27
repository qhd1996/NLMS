# -*- coding: utf-8 -*-
import time

import numpy as np
import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from code.util.config import Config
from code.models.gru_svm import GRU_SVM_Model

from code.util.util import load_data

print(os.listdir("../../input"))

data, label = load_data("../../input/test_data.npy")
data = data[:, :, np.newaxis]
label = label[:, np.newaxis]

test_x = data
test_label = label
print("Test.shape", test_x.shape, "Test label.shape", test_label.shape)

def test_model(model, test_x, test_label):
    start = time.time()
    test_x = torch.from_numpy(test_x).float()
    test_label = torch.from_numpy(test_label).float()

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
            preds_test[batch_start:batch_end, :] = np.argmax(out.detach().numpy(), axis=1)[:, np.newaxis]
        if (batch_end < test_x.shape[0]):
            x_batch = test_x[batch_end:, :, :]
            out = model(x_batch)
            preds_test[batch_end:, :] = np.argmax(out.detach().numpy(), axis=1)[:, np.newaxis]
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
    model = GRU_SVM_Model(config)
    model = model.cuda()
    test_model(model, test_x, test_label)
