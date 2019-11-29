import os
import time
import numpy as np
import torch 
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm
import matplotlib.pyplot as plt
import pandas as pd

from model import LSTMModel, savemodel, loadmodel

'''Method to train the LSTM Model
    Input: LSTM/RNN, optimizer, Lossfunction, Trainloader,  
    Validloader, number of epochs (int), learningrate (float), 
    batch_size (int), device (str) 
    directory to save checkpoint to (str)
    
    Output Console: Current Epoch with Trainingloss and Validationloss
    Output: Saves a state dictrionary of the Model in given directory'''
    
def train_model(model, optimizer, criterion, Trainloader, Validloader, epochs, learning_rate,batch_size,valid_batch_size, device, home_directory):
        t1 = time.time()
        print_every = 5
        counter = 0
        
        model.train()
        model.to(device)
        loss_graph = []
        testloss_graph = []
        for e in range(epochs): #Training goes epoch for epoch
           
            hidden = model.init_hidden(batch_size,device) # the hidden states need to be initialized every epoch with empty Cell- and Hiddenstates, otherwise the computational graph gets too big
            for inp, label in Trainloader: #Gets input and labelbatch from given Dataloader
                counter += 1
            
                inp = inp.permute(0,2,1) #Transforms from batch_sizeXInput_LengthXSequence_length to batch_sizeXSequence_lengthxInpt Length
                inp = inp.float()
                inp = inp[:,:,:-1] #Remove the RUL, since we dont know the true RUL
                label = label.permute(0,2,1) #Transforms from batch_sizeXInput_LengthXSequence_length to batch_sizeXSequence_lengthxInpt Length
                label = label.float()
                inp,label = inp.to(device),label.to(device)
          
                
                label = label.float().to(device)
                optimizer.zero_grad()
                hidden = (hidden[0].data, hidden[1].data) #isolate the hidden state from the computational graph
                out, hidden = model.forward(inp,hidden) #Forwardpass
    
                out = out.reshape(batch_size,-1,inp.shape[2]+1)
                loss = criterion(out,label)#Calculation of loss
                loss.backward()#backpropagation
                optimizer.step()#backpropagation

            #Doing the same for the Validation data, except backpropagation, to check the performance on unknown data
                if counter % print_every == 0:
                    model.eval()
                    hidden_eval = model.init_hidden(valid_batch_size,device)
                    with torch.no_grad():
                        for valid_inp, valid_label in Validloader:
                        
                            valid_inp = valid_inp.permute(0,2,1)
                            valid_label = valid_label.permute(0,2,1)
                            valid_inp = valid_inp.float()
                            valid_inp = valid_inp[:,:,:-1]
                            valid_label = valid_label.float()
                            
                            valid_inp, valid_label = valid_inp.to(device), valid_label.to(device)
                
                            model.to(device)
                            output, _ = model.forward(valid_inp,hidden_eval)
                            output = output.reshape(valid_batch_size,-1,valid_inp.shape[2]+1)
                            valid_label = valid_label.reshape(valid_batch_size,-1,valid_inp.shape[2]+1)
                            test_loss = criterion(output, valid_label)
                            print("{}/{}".format(e,epochs))
                            print(loss.cpu().data.numpy())
                            loss_graph.append(loss.detach().to("cpu").numpy())
                            print(test_loss.cpu().data.numpy())
                            testloss_graph.append(test_loss.detach().to("cpu").numpy())
                            model.train()
                    
                    
        t2 = time.time()
        print("The Calculation took",t2-t1, "seconds")
        savemodel(model) #saving the model state dictionary as .pth format (lookup at Model.py class)
        
        #plotting loss of training and validation            
        loss_graph = np.asarray(loss_graph)
        testloss_graph = np.asarray(testloss_graph)
        os.chdir(home_directory)
        plt.xlim(0,epochs/print_every)
        plt.ylim(0,0.05)
        plt.xlabel("Epochs[1/5]")
        plt.ylabel("Loss")
        plt.plot(loss_graph, label = 'Training loss')
        plt.plot(testloss_graph, label = 'Validation loss')
        plt.legend()
        plt.grid()
        plt.savefig("Loss")    
        plt.close()
        plt.clf()

