import os
import numpy as np
import torch 
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from features import get_data_from_path
from model import LSTMModel, savemodel, loadmodel
from training import train_model
from predict import predicting
from process import post_processing

'''
Dataloader that creates Databatch

Input = Numpy Array of Data
Output = Batching the Data'''
class TimeSeriesLoader(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(self.x)

    def __getitem__(self,index):
        return self.x[index], self.y[index] 
    
    def __len__(self):
        return self.len
    



if __name__ == '__main__':
    using_gpu = torch.cuda.is_available()
    #Checking if graphic card is able to process cuda operation
    if using_gpu:
        device = "cuda:2"
        print("Training on GPU")
    else: 
        device = "cpu"
        print("Training on CPU")
        
    
    os.path.dirname(os.path.abspath(__file__))
    
    #Hyperparameter for Models
    modelparameters = [34,128,35,2,1,0.2] #inputnNeurons, hidden hize, output neurons,layers, directions, dropout
    model = LSTMModel(modelparameters[0],modelparameters[1],modelparameters[2],modelparameters[3],modelparameters[4],modelparameters[5])#creating the model 
    model.to(device)
    
    #Hyperparameter of the Training
    batch_size = 6
    valid_batch_size = 11
    epochs = 500
    learning_rate = 0.001
    print_every = 5
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
    
    home_directory = os.path.dirname(os.path.abspath(__file__))
    
  
    training_path = ""
    valid_path = ""
    '''1. Checks if a .npy Dataset is in current directory
       2. If  there is a file it will load the file as numpy matrix
       2. If there isnt a file it will create a Trainset and save the matrix
       3. For creation of Dataset Features.get_data_from_path will be used(Input: Datasetdirectory, Output Numpy Matrix with Features)
       example for training_path and validation_path
       training_path ="/home/users/username/Desktop/Pronostia_LSTM/Femto_Bearing/Learning_set" 
       valid_path = "/home/users/username/Desktop/Pronostia_LSTM/Femto_Bearing/Full_Test_set" 
        '''
    print("Train_data is beeing prepared")
    
    training_file = home_directory+"/training_data_multi.npy"
    training_sequences = home_directory+"/sequences_training_data_multi.npy"
    train_data = np.empty(1)
    train_sequence = np.empty(1)
    if os.path.isfile(training_file):    
        train_sequence = np.load(training_sequences)
        train_data = np.load(training_file)
        print("train_data has been found")
    else:
        training_path = input("Enter the directory path of the training set ")
        train_data,train_sequences = get_data_from_path(path = training_path, name = "training_data_multi.npy")
        train_data = np.asarray(train_data)
        train_data = np.load(training_file)
        print("Train_data is ready")
    print("Train_data is prepared")
    print(train_data.shape)
    print(train_sequence)
    
    
    print("Validation_data is beeing prepared")
    validation_file = home_directory+"/valid_data_multi.npy"
    validation_sequences = home_directory+"/sequences_valid_data_multi.npy"
    valid_data = np.empty(1)
    valid_sequences = np.empty(1)
    if os.path.isfile(validation_file):
        valid_sequences = np.load(validation_sequences)
        valid_data = np.load(validation_file)
        print("valid_data has been found")
    else:
        valid_path = input("Enter the directory path of the validation set ")
        valid_data, valid_sequences = get_data_from_path(path = valid_path, name = "valid_data_multi.npy")
        valid_data = np.asarray(valid_data)
        valid_data = np.load(validation_file)
        print("valid_data is ready")
    print("Validation_data is prepared")
    print(valid_data.shape)
    print(valid_sequences)
    

    
    """Counting the Number of Bearings in each dataset"""
    train_folders_count = train_sequence.reshape(-1).shape[0]
    valid_folders_count = valid_sequences.reshape(-1).shape[0]
    
    
    #Deviding Set into input and label data. example: input goes from 0-999, label goes from 1-1000
    train_inp = train_data[:,:-1]
    train_label = train_data[:,1:]
    train_inp = train_inp.reshape(train_folders_count,-1,train_inp.shape[1])
    train_label = train_label.reshape(train_folders_count,-1,train_label.shape[1])
    
    valid_inp = valid_data[:,:-1]
    valid_label = valid_data[:,1:]
    valid_inp = valid_inp.reshape(valid_folders_count,-1,valid_inp.shape[1])
    valid_label = valid_label.reshape(valid_folders_count,-1,valid_label.shape[1])
    
    
    print(train_inp.shape)
    print(train_label.shape)
    print(valid_inp.shape)
    print(valid_label.shape)
    
    ''' Create Dataloder with TimeseriesLoader Class
    Input: Input (np array), Label (np array)
    Output: Batched np array
    '''
    train_inp = np.array(train_inp)
    train_label = np.array(train_label) 
    Trainset = TimeSeriesLoader(train_inp,train_label)
    
    
    valid_inp = np.array(valid_inp)
    valid_label = np.array(valid_label)
    Validset = TimeSeriesLoader(valid_inp,valid_label)

    '''Creating Generatorobject for Input and Label Data'''
    Trainloader = DataLoader(Trainset, batch_size, shuffle = False, drop_last=False, num_workers=0)
    Validloader = DataLoader(Validset, valid_batch_size, shuffle = False, drop_last=False, num_workers=0)
    
    
    
    #Training the model
    train_model(model, optimizer, criterion, Trainloader, Validloader, epochs, learning_rate, batch_size, valid_batch_size, device, home_directory)
    
    
    model = loadmodel(modelparameters) #Loading Model
    device = 'cuda:2'
    model.to(device)
    
    timesteps = 100
    feature = 34 #0RMS_X 1RMS_Y  2Kurtosis_x 3Kurtosis_y ... -3Wavelet_energy -2wiener Entropy -1RUL
    bearing = 9
    input_sequence = 2405 #the length of the padded input data
    
    #Calling the Predicting Method 
    predicting(model,Validset,timesteps,feature,bearing,device, home_directory,input_sequence)
    
    #Calling the below function post processes the predicted data and plots it
    post_processing(valid_sequences,Validset, bearing, feature)
    
    
'''sources and helpful links:
    https://github.com/osm3000/sequence_generation_pytorch
    https://lirnli.wordpress.com/2017/09/01/simple-pytorch-rnn-examples/
    https://github.com/albertlai431/Machine-Learning/blob/master/Text%20Generation/Shakespeare.py#L250
    https://pytorch.org/docs/stable/nn.html
    https://github.com/ngarneau/understanding-pytorch-batching-lstm
    '''