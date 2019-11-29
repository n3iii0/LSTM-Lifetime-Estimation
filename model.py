import numpy as np
import torch 
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm
import matplotlib.pyplot as plt
import pandas as pd

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers, num_directions, dropout):
        super(LSTMModel,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers
        self.num_directions = num_directions
        self.bi = False
        self.dropout = dropout
        
        #request for biderectional lstm
        if num_directions >= 2:
            self.bi = True
            print("Bidirectional LSTM")
        
        else:
            self.bi = False
            print("Standard LSTM")
            
        if self.layers <= 1:
            self.dropout = 0
            
        #creted model
        self.LSTM1 = nn.LSTM(self.input_size, self.hidden_size, self.layers, batch_first = True, bidirectional = self.bi, dropout = self.dropout) 
        self.linear1 = nn.Linear(self.hidden_size, self.output_size) 
        
        
    def forward(self, inp, hidden):
        batch_size = inp.size(0)
        seq_len = inp.size(1)
        output, hn_cn1 = self.LSTM1(inp, hidden) #inp = batch_size, seq_len, inp,size
        
        
        if self.bi == True:
            output = output.view(batch_size,seq_len, self.num_directions, self.hidden_size)
            output = output[:,:,0,:] #Bidirectional LSTM gives the Forward [0] and Backward[1] Hidden-State
        
        else:
            output = output.contiguous().view(-1,self.hidden_size)
    
        
        output = output.contiguous().view(-1,self.hidden_size) #output1 transform to seq_len*batch_size, hiddensize
        out = self.linear1(output) #hidden_size,batch_size*seq_leq
        out = out.view(batch_size,inp.size(1),-1)#batch_size,seq_len,1 get the last input as it is the next timestep
        #batch_size,1 letzten output der sequence nehmen
        return out, hn_cn1
    
    def init_hidden(self,batch_size,device):
        h01 = torch.zeros(self.layers * self.num_directions, batch_size, self.hidden_size, requires_grad =True).to(device) #Initializierung des leeren Hiddenstates und Cellstates 
        c01 = torch.zeros(self.layers * self.num_directions, batch_size, self.hidden_size, requires_grad = True).to(device)
        
        return (h01,c01)
    
#save model state dict as
def savemodel(model,path = "Checkpoint.pth"):
    model.to("cpu")
    torch.save(model.state_dict(), path)
    print('Model has been saved')

#load model stat dict from:
def loadmodel(modelparameters, path = "Checkpoint.pth"):
    print('Model is Loading')
    model = LSTMModel(modelparameters[0],modelparameters[1],modelparameters[2],modelparameters[3],modelparameters[4], modelparameters[5])
    model.load_state_dict(torch.load(path))
    print('Model has been loaded')
    return model