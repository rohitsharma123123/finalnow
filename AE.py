import pandas as pd
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class Autoencoder(torch.nn.Module):
    def __init__(self, n_in, n_hidden=10, sparsity_target=0.05, sparsity_weight=3, lr=0.01, weight_decay=0.001 , dropout = 0.25):#lr=0.0001):
        super(Autoencoder, self).__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        self.weight_decay = weight_decay
        self.lr = lr
        self.dropout = dropout
        #self.BETA = 3
        #self.rho = torch.FloatTensor([0.05 for _ in range(self.n_hidden)]).unsqueeze(0)
        self.rho = torch.tensor(0.05)
        self.kl_term = self.rho * torch.log(self.rho) + (1.0-self.rho) * torch.log(1.0-self.rho)
        self.build_model()
    # end constructor


    def build_model(self):
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.n_in, self.n_hidden),
            nn.Dropout(self.dropout),
            )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.n_hidden, self.n_in),
            torch.nn.Sigmoid())#,
            
        self.l1_loss = torch.nn.L1Loss(size_average=False)
        self.optimizer = opt.Adam(self.parameters(), self.lr, weight_decay=self.weight_decay)
    # end method


    def forward(self, inputs):
        hidden = self.encoder(inputs)
        rho_hat = torch.mean(hidden, dim=0, keepdim=False)
        #hidden_mean = torch.mean(hidden, dim=0)
        #sparsity_loss = torch.sum(self.kl_divergence(self.sparsity_target, hidden_mean))
        #sparsity_l =[]
        #spar_weight = []
        #for s_weight in self.sparsity_weight:
            #sparsity_loss = s_weight*(torch.sum(self.kl_divergence(self.rho , rho_hat)))
            #sparsity_l.append(sparsity_loss)
            #spar_weight.append(s_weight)
        #sparsity_loss = min(sparsity_l)
        #sparsity_loss = s_weight*(torch.sum(self.kl_divergence(self.rho , rho_hat)))
        #regularization = self.sparsityBeta * K.sum(kl_divergence(self.kl_term, self.p, p_hat))
        sparsity_loss = self.sparsity_weight*(torch.sum(self.kl_divergence(self.kl_term,self.rho , rho_hat)))
        #index_min = min(range(len(sparsity_l)), key=sparsity_l.__getitem__)
        #print(spar_weight[index_min])

        #sparsity_loss = self.BETA * torch.sum(self.kl_divergence(self.rho , rho_hat))
        #sparsity_loss = self.kl_divergence(self.sparsity_target , rho_hat)
        #sparsity_loss = self.kl_divergence(self.rho , rho_hat)

        return self.decoder(hidden), sparsity_loss
        # end method


    def kl_divergence(self,kl_term, p ,p_hat):
        #rho = torch.FloatTensor([0.01 for _ in range(self.n_hidden)]).unsqueeze(0)
        #rho_hat = torch.sum(encoded, dim=0, keepdim=True)
        #s1 = torch.sum(p * torch.log(p / q))
        #s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
        #return s1 + s2
        #return ((p * torch.log(p / q)) + ((1 - p) * torch.log((1 - p) / (1 - q)))) # Kullback Leibler divergence
        #return F.kl_div(p, q)
        return (kl_term - ((p * torch.log(1e-10 + p_hat)) - ((1.0-p) * torch.log(1e-10 + 1.0-p_hat))))
        # end method


    def fit(self, X, n_epoch=10, batch_size=60, en_shuffle=False):
        for epoch in range(n_epoch):
            if en_shuffle:
                print("Data Shuffled")
                #X = sklearn.utils.shuffle(X)
            for local_step, X_batch in enumerate(self.gen_batch(X, batch_size)):
                inputs = torch.from_numpy(X_batch.astype(np.float32))
                outputs, sparsity_loss = self.forward(inputs)

                l1_loss = self.l1_loss(outputs, inputs)
                l2_regularization = torch.tensor(0.03)
                for param in self.parameters():
                    l2_regularization += torch.norm(param)
                loss = l1_loss + sparsity_loss + l2_regularization
                self.optimizer.zero_grad()                             # clear gradients for this training step
                loss.backward(retain_graph = True)                                        # backpropagation, compute gradients
                self.optimizer.step()                                  # apply gradients
                #if local_step % 50 == 0:
                    #print ("Epoch %d/%d | Step %d/%d | train loss: %.4f | l1 loss: %.4f | sparsity loss: %.4f"
                           #%(epoch+1, n_epoch, local_step, len(X)//batch_size,
                             #loss.data[0], l1_loss.data[0], sparsity_loss.data[0]))
    # end method


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
