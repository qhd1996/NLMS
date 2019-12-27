import torch
import torch.nn as nn

class LinearSVM(nn.Module):

    def __init__(self, feature_dim, num_classes):
        super(LinearSVM, self).__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        h = self.linear(x)
        return h

# svm = LinearSVM(20, 2)
# # print(svm.linear.weight.shape)
# # result = (1 / 2 * torch.matmul(svm.linear.weight, svm.linear.weight.T)).sum()
# # print(result)
# result = torch.ones(4,2)
# print(-result)
# label = torch.ones(1,2)
# label[0, 1] = -1
# predict = torch.ones(1,2)
# predict[0, 0] = 1
# predict[0, 1] = -1
# print((1 - label * predict).sum())
# class_num = 2
# batch_size = 4
# label = torch.LongTensor(batch_size, 1).random_(0, 2)
# predict = torch.LongTensor(batch_size, 2).random_(-1, 1)
# print(label)
# print(predict)
# one_hot = -1 * torch.ones(batch_size, class_num)
# one_hot = one_hot.scatter_(1, label, 1)
# print(one_hot)
# out = torch.max(torch.ones(batch_size, class_num), 1 - one_hot.float() * predict.float())
# print(out)


