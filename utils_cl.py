import pandas as pd
import numpy as np
import pickle
import shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import sklearn
import time
import os
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def prepare_data_lstm_og(x_encoded, y_close, time_steps, log_return=True, train=True):
    ct = 0
    data = []
    for i in range(len(x_encoded)-time_steps):
        ct +=1
        if train:
            x_train = x_encoded[i:i+time_steps]
        else:
            x_train = x_encoded[:i+time_steps]

        data.append(x_train)

    if log_return==False:
        y_close = y_close.pct_change()[1:]
        #pass
    else:
        y_close = (np.log(y_close) - np.log(y_close.shift(1)))[1:] # the log return, i.e. ln(y_t/y_(t-1))

    if train:
        y = y_close[time_steps-1:]
    else:
        y=y_close

    return data, y


def prepare_data_lstm(x_encoded, y_close, time_steps, log_return=True, train=True):
    ct = 0
    data = []
    for i in range(len(x_encoded)-time_steps):
        ct +=1
        if train:
            x_train = np.vstack(x_encoded[i:i+time_steps])
        else:
            x_train = np.vstack(x_encoded[:i+time_steps])

        data.append(x_train)

    if log_return==False:
        y_close = y_close.pct_change()[1:]
        #pass
    else:
        y_close = (np.log(y_close) - np.log(y_close.shift(1)))[1:] # the log return, i.e. ln(y_t/y_(t-1))

    if train:
        y = y_close[time_steps-1:]
    else:
        y=y_close

    return data, y

def prepare_data_lstm_cl(x_encoded, y_close, time_steps, log_return=True, train=True):
    ct = 0
    data = []
    for i in range(len(x_encoded)-time_steps):
        ct +=1
        if train:
            x_train = np.vstack(x_encoded[i:i+time_steps])
        else:
            x_train = np.vstack(x_encoded[:i+time_steps])

        data.append(x_train)

    if log_return==False:
        y_close = y_close.pct_change()[1:]
        #pass
    else:
        #y_close = (np.log(y_close) - np.log(y_close.shift(1)))[1:] # the log return, i.e. ln(y_t/y_(t-1))
        y_close = ((y_close) - (y_close.shift(1)))[1:]

    if train:
        y = y_close[time_steps-1:]
    else:
        y=y_close
        
    y_ = np.where(y < 0.0, 0, 1)
        
    #if y<0.0:
     #   y_ = 'negative'
    #else:
     #   y_ = 'positive'

    return data, y_


class ExampleDataset(Dataset):

    def __init__(self, x, y, batchsize):
        self.datalist = x
        self.target = y
        self.batchsize = batchsize
        self.length = 0
        self.length = len(x)

    def __len__(self):
        return int(self.length/self.batchsize+1)

    def __getitem__(self, idx):
        x = self.datalist[idx*self.batchsize:(idx+1)*self.batchsize]
        y = self.target[idx*self.batchsize:(idx+1)*self.batchsize]
        sample = {'x': x, 'y': y}

        return sample


def evaluate_lstm_cl(dataloader, model, criterion):

    pred_val = []
    target_val = []
    model.eval()
    # do evaluation
    loss_val = 0
    sample_cum_x = [None]
    #pred = numpy.array(1000,2)
    #targ = numpy.array(1000,2)
    p =  np.zeros((500, 2))
    t =  np.zeros((500, 2))

    for j in range(len(dataloader)):

        sample = dataloader[j]
        sample_x = sample["x"]
        sample_y = torch.tensor(sample["y"])
        #p = np.array(500,2)
        #t = np.array(500,2)

        if len(sample_x) != 0:

            sample_x = np.stack(sample_x)
            input = Variable(torch.FloatTensor(sample_x), requires_grad=False).cuda()
            input = torch.transpose(input, 0, 1).cuda()
            #target = Variable(torch.FloatTensor(sample["y"].as_matrix()), requires_grad=False)
            #target = torch.FloatTensor([x for x in sample["y"]])
            target = nn.functional.one_hot(sample_y.to(torch.int64),num_classes=2).cuda()
            target = target.to(dtype=torch.float32).cuda() #dtype = torch.float32
            #print('in evaluate lstm')
            #print(input.shape)
            #print(target.shape)

            out = model(input)

            loss = criterion(out, target)

            #loss_val += float(loss.data.numpy())
            #pred_val.extend(out.data.numpy().flatten().tolist())
            #target_val.extend(target.data.numpy().flatten().tolist())
            values,indices = torch.max(out,0)
            
            #loss_val += float(loss.cpu().data.numpy())
            loss_val += float(loss.cpu().item())
            #p = np.append(out.cpu().data)
            #t = np.append(target.cpu().data)
            pred_val.extend(out.cpu().data.numpy().tolist())
            target_val.extend(target.cpu().data.numpy().tolist())

    return loss_val, pred_val, target_val

def evaluate_lstm_deap(dataloader, model, criterion):

    pred_val = []
    target_val = []
    model.eval()
    # do evaluation
    loss_val = 0
    sample_cum_x = [None]

    for j in range(len(dataloader)):

        sample = dataloader[j]
        sample_x = sample["x"]
        sample_y = torch.tensor(sample["y"])

        if len(sample_x) != 0:

            sample_x = np.stack(sample_x)
            input = Variable(torch.FloatTensor(sample_x), requires_grad=False)
            input = torch.transpose(input, 0, 1)
            #target = Variable(torch.FloatTensor(sample["y"].as_matrix()), requires_grad=False)
            #target = torch.FloatTensor([x for x in sample["y"]])
            target = nn.functional.one_hot(sample_y)
            target = target.to(dtype=torch.float32) #dtype = torch.float32

            out = model(input)

            loss = criterion(out, target)

            loss_val += float(loss.data.numpy())
            pred_val.extend(out.data.numpy().flatten().tolist())
            target_val.extend(target.data.numpy().flatten().tolist())

    return loss_val,



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', name="checkpoint"):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(name) + 'model_best.pth.tar')