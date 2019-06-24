import pandas as pd
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn.functional as F
from torch.autograd import Variable


class Lstm(nn.Module):
    def __init__(self, nb_features=1, hidden_size=100, nb_layers=5, dropout=0.25,bs = 60 ,act = 'relu'):
        super(Lstm, self).__init__()
        self.nb_features=nb_features
        self.hidden_size=hidden_size
        self.nb_layers=nb_layers
        self.lstm = nn.LSTM(self.nb_features, self.hidden_size, self.nb_layers, dropout=dropout)
        #self.rel = nn.ReLU()
        #self.prel = nn.PReLU(num_parameters=1, init=0.25)
        self.leak = nn.LeakyReLU()
        self.tan = nn.Tanh()
        self.rel = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.lin = nn.Linear(self.hidden_size,1)
        self.act = act
        self.init_hidden(bs)

    def forward_states(self, inputs):
        x = inputs
        #hx, cx = hn
        flag=True #to flag when at start of time series

        if self.flag:
            cx = torch.zeros(self.nb_layers, input.size()[1], self.hidden_size)
            hx = torch.zeros(self.nb_layers, input.size()[1], self.hidden_size)
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)
        #h0 = torch.zeros(self.nb_layers, input.size()[1], self.hidden_size)
        #c0 = torch.zeros(self.nb_layers, input.size()[1], self.hidden_size)
        hx, cx = self.lstm(x, (hx, cx))
        relu_out = self.rel(hx[-1])
        out = self.lin(relu_out)
        return out, (hx, cx)
        #return out
        
    def forward_raw(self, input):
        h0 = Variable(torch.zeros(self.nb_layers, input.size()[1], self.hidden_size))
        #print(type(h0))
        c0 = Variable(torch.zeros(self.nb_layers, input.size()[1], self.hidden_size))
        #print(c0.shape)
        #print(type(c0))
        output, hn = self.lstm(input, (h0, c0))
        #output = F.relu(o)
        out = self.lin(output[-1])
        return out
    
    def forward(self, input):
        #bs = input[0].size(0)
        #if self.h[0].size(1) != bs:
            #self.init_hidden(bs)
        #output,h = self.lstm(input, self.h)
        #print(c0.shape)
        #print(type(c0))
        #output, self.h = self.lstm(input, self.h)
        #output = F.relu(output)
        #output = self.prel(output)
        #output = self.leak(output)
        #out = self.lin(output[-1])
        h0 = Variable(torch.zeros(self.nb_layers, input.size()[1], self.hidden_size).cuda())
        #print(type(h0))
        c0 = Variable(torch.zeros(self.nb_layers, input.size()[1], self.hidden_size).cuda())
        #print(type(c0))
        output, hn = self.lstm(input, (h0, c0))
        if self.act == 'relu':
            output = self.rel(output)
        elif self.act == 'sigmoid':
            
            output = self.sig(output)
        else:
            output = self.tan(output)   
        

        #output = self.leak(output)
        #output = F.relu(self.lin(output))
        out = self.lin(output[-1])
        return out
        #return out
    
    def init_hidden(self, bs):
        self.h = (Variable(torch.zeros(self.nb_layers, bs, self.hidden_size)),
                  Variable(torch.zeros(self.nb_layers, bs, self.hidden_size)))
                  
    def repackage_var(h):
        return Variable(h.data) if type(h) == Variable else tuple(repackage_var(v) for v in h)
    
class Lstm_Model(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Lstm_Model, self).__init__()
        self.lstm = nn.LSTMCell(10, hidden_size)
        self.rel = nn.ReLU()
        self.lin = nn.Linear(hidden_size,1)
    def forward(self, inputs):  #and then in def forward:
        x, (hx, cx) = inputs
        #print(x.shape)
        #x = x.view(x.size(0), -1)
        out = []
        cn = []
        for i in range(len(x)):
            print(x[i].shape)
            hx, cx = self.lstm(x[i], (hx[i], cx[i]))
            out.append(hx)
            cn.append(cx)
        #hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return x, (hx, cx)
