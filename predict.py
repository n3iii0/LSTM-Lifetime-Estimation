import os
import numpy as np
import torch 
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm
import matplotlib.pyplot as plt
import pandas as pd

from model import LSTMModel, savemodel, loadmodel

"""Class for Predicting the upcoming timesteps for the PRONOSTIA DATA
    Input: an LSTMModel, Dataset, number of timesteps that you want to predict(int), 
           number of feature you want to analyze(int), number of bearing(int), 
           device, the directory you want to save the results(str), length of Input sequence(int)
    Output: Plot of the Feature and Bearing, Results array in .npy format
    Output Console: Showing how long the current inp_sequence is""" 


def predicting(model,Validset,timesteps,feature,bearing, device, home_directory,input_sequence = 2000):    

    hidden = model.init_hidden(1,device)
    inp,label = Validset[bearing]    
    
    inp = torch.tensor(inp)
    inp = inp.unsqueeze(2)
    inp = inp.permute(2,1,0).float()
    
    root_inp = inp[:,:input_sequence,:]
    whole_seq = inp[:]
    inp = inp[:,:input_sequence,:]
    inp = inp.to(device)
    
    #Performas a Forwordpropagation for each timestep
    #the predicted output will be attached to the input sequence
    #Repat Forwardpropagation with new input sequence
    
    for i in range(timesteps):    
        inp = torch.tensor(inp)
        pred_inp = inp[:,:,:-1] #Taking the RUL out of the Input, because we don't know it during Prediction
        pred_inp = pred_inp.float()
        pred_inp = pred_inp.to(device)
        print(pred_inp.shape)
        hidden = (hidden[0].data,hidden[1].data)
        out, hidden = model.forward(pred_inp,hidden)
        y = out[:,-1,:].unsqueeze(0)
        inp = torch.cat((inp,y), dim = 1)
        y = y.to("cpu")
        
    
    results = inp.to('cpu').detach().numpy()
    np.save("input_result.npy", results)
    
    out_plot = out.to('cpu').detach().numpy()
    np.save("output_results.npy", out_plot)
    out_plot = out_plot[:,:,feature].reshape(-1)
    
    whole_seq = whole_seq.to('cpu').detach().numpy()
    whole_seq_plot = whole_seq[:,:,feature].reshape(-1)
    
    #inp_plot = inp_plot[:,:,feature].reshape(-1)
    
    root_inp = root_inp.to('cpu').detach().numpy()
    root_inp = root_inp[:,:,feature].reshape(-1)
    
    os.chdir(home_directory)
    plt.plot(whole_seq_plot, label = "Actual RUL")
    plt.plot(root_inp, label = "Input sequence")
    #plt.plot(inp_plot)
    plt.plot(out_plot, label = "Predicting")
    
    plt.grid()
    plt.legend()
    plt.savefig("Output_plot")
    plt.close()
