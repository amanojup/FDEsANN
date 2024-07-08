#!/usr/bin/env python
# coding: utf-8

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from scipy.special import gamma
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
import torch
import argparse

# set random seed
np.random.seed(42)
torch.set_printoptions(precision=10)

def g(x):
    '''
    inputs
        :param x: float, x value
    return
        concrete example functions value
    e.x.
    ===================
    x + 1
    '''
    return 1


def d(x, i, alpha):
    '''
    \Gamma function
    '''
    if i < np.ceil(alpha):
        return 0.0
    return math.gamma(i + 1) / math.gamma(i + 1 - alpha) * x ** (i - alpha)


def f(x, a_value, n=3):
    '''
    left formula, Df(x) : spde
    inputs
        :param x: float, x value
        :param a_value: train args
        :param n:  a_{j}, j=1,2,...,n
    '''
    global lambda_v
    # artificially set parameters
    lambda_v = 0.88

    error = torch.zeros(1)
    for idx, xx in enumerate(x): 
        total_value = 0
        for i in range(n+1):
            total_value += a_value[i] * d(xx, i, 1)

        for i in range(n+1):
            total_value += a_value[i] * (xx/2) **i
        total_value =  2*total_value**2    

        g_result = g(xx)

        er = (total_value - g_result) ** 2 + lambda_v * (a_value[0] - 0) ** 2 + lambda_v * (a_value[0] - 1) ** 2
        error += er

    return torch.sqrt(error / (len(x) + 1))

class DataSet(object):
    def __init__(self, sample_list):
        self.len = len(sample_list)
        self.data = torch.from_numpy(np.array(sample_list, np.float32))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

# ANN
class MyNet(nn.Module):
    def __init__(self, input_param, hidden_num, output_param):
        super(MyNet, self).__init__()
        self.linear0 = nn.Linear(input_param, hidden_num, bias=True)
        self.linear1 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.linear2 = nn.Linear(hidden_num, hidden_num3, bias=True)
        self.linear3 = nn.Linear(hidden_num3, output_param, bias=True)

    def forward(self, x):
        x = self.linear0(x)
        x = F.relu(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.dropout(x, p=0.01)
        x = F.relu(x)
        x = self.linear3(x)
        return x
#axis    

data_nums = 11
# learning rate
lr = 0.001

input_param = data_nums
hidden_num = 30
hidden_num3 = 3
output_param = 4

# train nums
n_iter_max = 500
x = np.linspace(0, 1, data_nums)
#y = g(x)
# array2tensor
x_tensor = torch.from_numpy(x)
#y_tensor = torch.from_numpy(y)

dataset = DataSet(x)

train_loader = DataLoader(
    dataset=dataset,
    batch_size=data_nums,
    shuffle=True,
    num_workers=0
)
model = MyNet(input_param, hidden_num, output_param)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
model = model.float()
for iter_n in range(n_iter_max):
    for i, data in enumerate(train_loader, 0):
        a_value = model(data)
        loss = f(data, a_value, n=output_param - 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (iter_n) % 64 == 0:
        print("iter-->",iter_n, "loss-->", loss.data)
print(a_value)
print(x_tensor.float())
#total_value = 1
#n=4
#for j in range(len(data)):
 #  for i in range(n+1):
  #   outputs = a_value[i] * data[j] ** i
   #  total_value += outputs
   #print(total_value)
n=3
for idx, xx in enumerate(data):
    total_value = 0

    for i in range(n+1):
        total_value += a_value[i] * xx **i
    print(total_value)
    print(g(xx))
#print(outputs) 
u_ana = np.sin(data)
plt.figure()
plt.plot(x, total_value, label='ANN-based solution')
plt.plot(data, u_ana, '.', label='analytical solution')
plt.ylabel('u')
plt.xlabel('t')
plt.title('comparing solutions')
plt.legend()
plt.show()