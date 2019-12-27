# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

model = "GRU-Softmax"
df = pd.read_excel("results.xlsx", sheet_name = "softmax-loss")
iterations = []
for i in range(10):
    iterations.append(i + 1)
lr4 = df['lr1e-4']
lr5 = df['lr1e-5']
lr6 = df['lr1e-6']



#开始画图
plt.title('Learning rate and train loss for ' + model)
plt.plot(iterations, lr4, color='red', label='1e-4')
plt.plot(iterations, lr5, color='green', label='1e-5')
plt.plot(iterations, lr6, color='blue', label='1e-6')
plt.legend() # 显示图例

plt.xlabel('iteration times')
plt.ylabel('loss')
plt.savefig(model + ".png")
plt.show()
