import os
import glob
import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset 
import csv
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import pywt
import pywt.data
from scipy import signal, stats
from scipy.signal import savgol_filter

"""This Class prepares the Datasets from csv Files of the directories given by the PRONOSTIA DATASET
the function that uses all of the function in this class is named "get_data_from_path()"""

"""
Input: Path of the Folder (str)
Output: Dataset as np array"""

class TimeSeriesLoader(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(self.x)
        print(x.shape)

    def __getitem__(self,index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len


'''Function to get the Folder which is currently used
Input foldernumber (int), path of the Set (str)
Output path that opens folders one by one'''
def get_folder(foldernumber, path = "/home/users/username/Masterarbeit/Pronostia_LSTM/Training_set"):
    folders = os.listdir(path)
    folder = folders[foldernumber]
    folder_path = path + "/" + folder
    return folder_path


'''Loads the acc files one by one.
Input: accnumber(str) ,path of the current bearing(str)
output: pandas dataframe of one csv file'''
def get_accfile(filenumber,folder_path):
    os.chdir(folder_path)
    file_list = glob.glob("*.csv")
    
    delimiter = "" 
    
    with open (file_list[filenumber], 'r') as f: #checks if the csv is seperated by ',' or ';'
        dialect = csv.Sniffer().sniff(f.readline())
        f.seek(0)
        delimiter = dialect.delimiter
    
    acc_data = pd.read_csv(file_list[filenumber], header = None, delimiter = delimiter)
   
    return acc_data


'''gets the Acceleration of an Pandasfile
Input: Pandas Dataframe
Output: Numpy Array with x-y acceleration'''
def get_acceleration(acc_data):
    acc_horiz = acc_data.iloc[:,[4]]
    acc_vert = acc_data.iloc[:,[5]]

    acc_horiz = acc_horiz.to_numpy()
    acc_vert = acc_vert.to_numpy()
    return acc_horiz, acc_vert

'''Passes the acceleration or wavelet array and calculates RMS'''
def root_mean_square(data):
    X = data
    length = X.size
    sum = np.sum(data**2)
    return np.sqrt(sum/length)

'''Passes the acceleration or wavelet array and calculates Energy'''
def energy(data):
   
    E = np.absolute((data))**2
    E = np.sum(E)
    return E

'''Passes the acceleration calculates the Fast Fourier Transform (not used)'''
def calculate_fft(data):
    yf = fft(data)
    yf = abs(yf)
    return yf

'''Passes the acceleration and calculates Wiener Entropy'''
def wiener_entropy(data, f=1.0, minimum = 1e-12 ):
    
    _, power_spectrum = signal.welch(data)
    power_spectrum = np.maximum(power_spectrum,minimum)
    length = power_spectrum.size
    

    log_data = np.log(power_spectrum)
    log_data_sum = log_data.sum()/length
    geomMean = np.exp(log_data_sum)
    

    sum = power_spectrum.sum()
    aritmeticMean = sum/length
    
    wiener_entropy = geomMean/aritmeticMean
   
    return wiener_entropy

'''Passes the acceleration or wavelet array and calculates Kurtosis'''
def calculate_kurtosis(data):
    kurtosis = stats.kurtosis(data, axis = 0, fisher = False, bias = False)
    kurtosis = np.asarray(kurtosis)
    return kurtosis

'''Passes the acceleration or wavelet array and calculates Skewness'''
def calculate_skewness(data):
    skewness = stats.skew(data,axis = 0, bias = False)
    return skewness

'''Passes the acceleration or wavelet array and calculates Variance'''
def calculate_variance(data):
    variance = np.var(data)
    return variance

'''Passes the acceleration or wavelet array and calculates Peak to Peak Value'''
def peak_to_peak(data):
    ptp = np.ptp(data, axis = 0)
    return ptp

'''Passes the acceleration or wavelet array and calculates Impulsefactor'''
def impulse_factor(data):
    impulse_factor = np.max(np.absolute(data))/(np.mean(np.absolute(data)))
    return impulse_factor

'''Passes the acceleration or wavelet array and calculates margin factor'''
def margin_factor(data):
    mf = np.max(np.absolute(data))/(root_mean_square(data))
    return mf

'''Passes the acceleration or wavelet array and calculates wave factor'''
def wave_factor(data):
    data = np.absolute(data)
    wave_factor = np.sqrt(np.mean(data))/(np.mean(data))
    return wave_factor

'''Passes the acceleration or wavelet array and calculates wave standard_derivation'''
def standard_derivation(data):
    std = np.std(data)
    return std

'''Passes the acceleration or wavelet array and calculates variation coefficient'''
def variation_coefficient(data):
    vc = np.std(data)/np.mean(data)
    vc = np.nan_to_num(vc) #in case mean is at zero, the vector returns nan values which have to be replaced by 0
    return vc

'''Passes the acceleration or wavelet array and calculates mean'''
def mean(data):
    return np.mean(data)

'''Passes the acceleration or wavelet array and calculates maximum'''
def maximum(data):
    return np.max(data)

'''Passes the acceleration or wavelet array and calculates absolute average'''
def absolute_average(data):
    absolute = np.abs(data)
    absolute_average = np.mean(absolute)
    return absolute_average

'''Reads the bearing number from the path and returns a string with the it'''
''' Input: Folderpath(str)'''
''' Output: Bearingnumber(str)'''
def get_bearing_number(path):
    bearing_path = os.path.dirname(path)
    bearing = os.path.basename(bearing_path)
    bearing = bearing.replace("Bearing", "")
    return bearing

'''Passes the acceleration or wavelet array and applys wavelettransform
a = data after low pass filter
d = data after high pass filter
aad = signal passed two times the low pass filter than a high pass filter
'''
def wavelet_transform(data, wavelet = 'db10', level = 3):
         wp = pywt.WaveletPacket(data=data, wavelet= wavelet, mode='symmetric', maxlevel=level)
         x = wp['aad'].data
         
         return x


def get_remaining_RUL(bearing,path):
    '''
    returns remaining RUL for given (PRONOSTIA) bearing number
    :input: bearing number(str), bearing path
    :return: remaining RUL in [s]
    '''
    bearing = float(bearing.replace("_","."))

    #print(bearing)
    assert bearing in [1.1,1.2,2.1,2.2,3.1,3.2,1.3,1.4,1.5,1.6,1.7,2.3,2.4,2.5,2.6,2.7,3.3], 'no data exists for input'
    c=1
    
    if 'Full' or 'Test' in path: #If the training set and or the Full test is used the remaining RUL is 0
        RUL_dict = {'1.3': 0,
                    '1.4': 0,
                    '1.5': 0,
                    '1.6': 0,
                    '1.7': 0,
                    '2.3': 0,
                    '2.4': 0,
                    '2.5': 0,
                    '2.6': 0,
                    '2.7': 0,
                    '3.3': 0,
                    '1.1': 0,
                    '1.2': 0,
                    '2.1': 0,
                    '2.2': 0,
                    '3.1': 0,
                    '3.2': 0}
        
        return RUL_dict[str(bearing)]*c
    else:
        RUL_dict = {'1.3': 5730,
                    '1.4': 339,
                    '1.5': 1610,
                    '1.6': 1460,
                    '1.7': 7570,
                    '2.3': 7530,
                    '2.4': 1390,
                    '2.5': 3090,
                    '2.6': 1290,
                    '2.7': 580,
                    '3.3': 820,
                    '1.1': 0,
                    '1.2': 0,
                    '2.1': 0,
                    '2.2': 0,
                    '3.1': 0,
                    '3.2': 0}
        return RUL_dict[str(bearing)]*c
    
    

def get_last_timestamp(bearing):
    ''' gets the last timestap of the last csv'''
    '''Input Bearingnumber(str)'''
    '''Return last timestamp from last acc file of a bearing'''
    bearing = float(bearing.replace("_","."))

    assert bearing in [1.1,1.2,2.1,2.2,3.1,3.2,1.3,1.4,1.5,1.6,1.7,2.3,2.4,2.5,2.6,2.7,3.3], 'no data exists for input'
    c = 0.1
    time_dict = {'1.3': 1.6562e5,
                '1.4': 5.25e5,
                '1.5': 3.125e5,
                '1.6': 8.75e5,
                '1.7': 6.2499e4,
                '2.3': 6.7187e5,
                '2.4': 5.625e5,
                '2.5': 2.5e5,
                '2.6': 7.1875e5,
                '2.7': 2.5e5,
                '3.3': 3.125e5,
                '1.1': 1.6562e5,
                '1.2': 2.9687e5,
                '2.1': 9.8437e5,
                '2.2': 6.4062e5,
                '3.1': 2.1876e5,
                '3.2': 7.8124e4}
    
    return time_dict[str(bearing)]*c



'''Counts Files/Timeframes for a folder'''
'''Input: Bearingnumber (str), path of Bearingfolder(str)
Return: count of files in a folder(int)'''
def get_filecount(bearing, path):
    
    folder_path = path + "/Bearing" + str(bearing)
    os.chdir(folder_path)
    file_list = glob.glob("*.csv")
    acc_file_list = [x for x in file_list if "acc" in x]
    length = len(acc_file_list)
    

    return length

'''Calculates  the lineal regression of a bearing'''
'''Input bearing number(str), acc_number of file(int), path of bearing (str)'''
def get_current_RUL(bearing_number,acc_number,path):

    remaining_RUL = get_remaining_RUL(bearing_number,path)
    file_count = get_filecount(bearing_number,path)
    #print(file_count)
    if remaining_RUL == 0:
        m = -100/file_count
        current_RUL = m*acc_number+100
        if current_RUL <= 0:
            return 0
        return current_RUL
    
    else: 
        wear_rate = get_wear_rate(bearing_number,path)
        current_RUL = wear_rate*acc_number+100
        if current_RUL <= 0:
            return 0
        return current_RUL
    
''' calculates the wear rate if the RUL is unknown
contructing a slope for validation data not necessary for Full test set!'''
  
def get_wear_rate(bearing, path):
    last_timestamp = get_last_timestamp(bearing)
    remaining_RUL = get_remaining_RUL(bearing,path)
    forcasted_end = last_timestamp+remaining_RUL

    m = -100/forcasted_end

    #converting from seconds to file number
    filecount = get_filecount(bearing,path)
    relative_remaining_RUL = (m*last_timestamp+100)
    print(relative_remaining_RUL)
    wear_rate = (100-relative_remaining_RUL)/filecount

    return -wear_rate
    
def plot_results(data,name):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    plt.plot(data)
    plt.savefig(name)
    plt.close

'''Proceeds zero padding at the end of data'''
'''Input: numpay array of Features
   return: numpay array with post padded data'''
   
def post_padding_multiple(data):
    finished_vector = np.zeros((len(data)*len(data[0]),max(len(x[0]) for x in data)))
    for i,j in enumerate(data):
        #print(finished_vector[len(training_data[0])*i:i*len(training_data[0])+len(training_data[0])][0:len(training_data[i][0])].shape)
        finished_vector[j.shape[0]*i:i*j.shape[0]+j.shape[0],0:j.shape[1]] = j
        
    return finished_vector

'''Proceeds zero padding at the beginning of data (better than post padding)'''
'''Input: numpay array of Features
   return: numpay array with post padded data'''

def pre_padding_multiple(data):
    finished_vector = np.zeros((len(data)*len(data[0]),max(len(x[0]) for x in data)))
    for i,j in enumerate(data):
        #print(finished_vector[len(training_data[0])*i:i*len(training_data[0])+len(training_data[0])][0:len(training_data[i][0])].shape)
        finished_vector[j.shape[0]*i:i*j.shape[0]+j.shape[0],-j.shape[1]:] = j
        
    return finished_vector


'''applys Min Max Scaling on Data for each feature one by one'''
'''Input: List with every feature of a bearing folder
   return: a numpy array with scaled values'''
   
def scaling_single(data):
    for i,bearing in enumerate(data):
        for y in range(len(data[0])):
            maximum = np.max(data[i][y])
            minimum = np.min(data[i][y])
            
            data[i][y] = (data[i][y]-minimum)/(maximum-minimum)
    
    return data
        
'''applys Min Max Scaling on whole Data. Every Feature gets scaled by the same value!'''
'''Input: List with every feature of a bearing folder
   return: a numpy array with scaled values'''
def scaling_multiple(data):
    maximum = 0
    minimum = 0
    for i in range(len(data[0])):    
        maximum = max(np.max(x[i]) for x in data)
        minimum = min(np.min(x[i]) for x in data)
        for y ,x in enumerate(data):
            data[y][i] = (data[y][i]-minimum)/(maximum-minimum)
    return data


'''applys savgol_filter on whole Data. Not recommended for LSTMs use wavelettransforms for filter'''
'''Input: numpay array of Features
   return: a numpy array with filtered values'''
def filtering(data):
    for i in range(len(data)):
        print("here")
        print(data[i].shape[0])
        for j in range(data[i].shape[0]):
            data[i] = savgol_filter(data[i],101,2)   
    return data

'''counts how man csv(timeframes) are in each folder
input: numpy array of features before the padding
returns: list of sequences with count of every data'''
def get_sequencelist(data):
    sequence_list = []
    for i, bearing in enumerate(data):
        sequence_list.append(len(data[i][0]))
    return np.asarray(sequence_list)-1
        

'''This function calls most the function above
1. it opens every folder in a given path, 
2. reads every csv one by one after opening a folder
3. creates features
4. Min max scales them
5. Adds Zeropadding to the Data
6. Saves the Dataset as a .npy file'''

def get_data_from_path(path,name):  
    folderlist = os.listdir(path)
    folderlist.sort()
    features_vector = []
    training_data = []
    temp_vector = []
    #acc_vector_x = []
    for i, folder in enumerate(folderlist):
        folder_path = get_folder(i,path)
        #print(folder)
        os.chdir(folder_path)
        acc_file_list = glob.glob("*.csv")
        for i, acc_file in enumerate(acc_file_list):
            if "acc" in acc_file:
                print(acc_file)
                acc_data = get_accfile(i,folder_path)
                acc_x, acc_y = get_acceleration(acc_data)  
                acc_x = acc_x.ravel()
                wavelet_x = wavelet_transform(acc_x)
                acc_y = acc_y.ravel()
                wavelet_y = wavelet_transform(acc_y)
                #acc_vector = np.append([acc_x],[acc_y], axis = 0)
                #acc_vector_x.append(acc_x)
                
                rms_x = np.asarray(root_mean_square(acc_x))
                rms_y = np.asarray(root_mean_square(acc_y))
                #wavelet_rms_x = root_mean_square(wavelet_x)
                #wavelet_rms_y = root_mean_square(wavelet_y)
                features_vector.append(rms_x)
                features_vector.append(rms_y)
                #features_vector.append(wavelet_rms_x)
                #features_vector.append(wavelet_rms_y)
                
                kurtosis_x = calculate_kurtosis(acc_x)
                kurtosis_wavelet_x = calculate_kurtosis(wavelet_x)
                kurtosis_y = calculate_kurtosis(acc_y)
                kurtosis_wavelet_y = calculate_kurtosis(wavelet_y)
                features_vector.append(kurtosis_x)
                features_vector.append(kurtosis_y)
                features_vector.append(kurtosis_wavelet_x)
                features_vector.append(kurtosis_wavelet_y)
            
                margin_x = margin_factor(acc_x)
                margin_y = margin_factor(acc_y)
                features_vector.append(margin_x)
                features_vector.append(margin_y)
                
                variance_x = calculate_variance(acc_x)
                variance_y = calculate_variance(acc_y)
                features_vector.append(variance_x)
                features_vector.append(variance_y)
                
                std_x = standard_derivation(acc_x)
                std_y = standard_derivation(acc_y)
                features_vector.append(std_x)
                features_vector.append(std_y)
                
                
                vc_x = variation_coefficient(acc_x)
                vc_y = variation_coefficient(acc_y)
                features_vector.append(vc_x)
                features_vector.append(vc_y)
                
                
                skewness_x = calculate_skewness(acc_x)
                skewness_y = calculate_skewness(acc_y)
                features_vector.append(skewness_x)
                features_vector.append(skewness_y)
                
                
                ptp_x = peak_to_peak(acc_x)
                ptp_y = peak_to_peak(acc_y)
                features_vector.append(ptp_x)
                features_vector.append(ptp_y)
                
                
                impulse_factor_x = impulse_factor(acc_x)
                impulse_factor_y = impulse_factor(acc_y)
                features_vector.append(impulse_factor_x)
                features_vector.append(impulse_factor_y)
                
                
                WE_x = wiener_entropy(acc_x)
                WE_y = wiener_entropy(acc_y)
                features_vector.append(WE_x)
                features_vector.append(WE_y)
                
                aa_x = absolute_average(acc_x)
                aa_y = absolute_average(acc_y)
                features_vector.append(aa_x)
                features_vector.append(aa_y)
                
                
                maximum_x = maximum(acc_x)
                maximum_y = maximum(acc_y)
                features_vector.append(maximum_x)
                features_vector.append(maximum_y)
                
                
                
                mean_x = mean(acc_x)
                mean_y = mean(acc_y)
                wavelet_mean_x = mean(wavelet_x)
                wavelet_mean_y = mean(wavelet_y)
                features_vector.append(mean_x)
                features_vector.append(mean_y)
                features_vector.append(wavelet_mean_x)
                features_vector.append(wavelet_mean_y)
                
                
                wave_factor_x = wave_factor(acc_x)
                wave_factor_y = wave_factor(acc_y)
                features_vector.append(wave_factor_x)
                features_vector.append(wave_factor_y)
                
                wavelet_energy_x = energy(wavelet_x)
                wavelet_energy_y = energy(wavelet_y)
                features_vector.append(wavelet_energy_x)
                features_vector.append(wavelet_energy_y)
                
                
                bearing_number = get_bearing_number(folder_path + "/" + acc_file)
                RUL = get_current_RUL(bearing_number,i,path)
                print(RUL)
                features_vector.append(RUL)
                

                features_vector = np.asarray(features_vector).reshape(1,-1) #this list contains all features of a csv file
                temp_vector.append(features_vector.reshape(-1)) #this numpy array contains the complete feature data of a bearing folder
               
                features_vector = []
                #print(kurtosis.shape)
                #print(len(feature_vector))
        
        
        temp_vector = np.asarray(temp_vector)
        print(temp_vector.shape)
        #temp_vector[:,-1] = np.flip(temp_vector[:,-1])
        temp_vector = np.transpose(temp_vector)
        print(temp_vector.shape)
        training_data.append(temp_vector) #This matrix contains every Feature of every bearing folder
        temp_vector = []
        
    
    
    sequence_name = "sequences"+ "_" + name
    sequence_list = get_sequencelist(training_data) 
    print(sequence_list)
    scaled_vector = scaling_single(training_data)
    finished_vector = pre_padding_multiple(scaled_vector)
   
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(os.path.dirname(os.path.abspath(__file__)))
    np.save(name,finished_vector,allow_pickle = True) #saves a Trainingset,
    np.save(sequence_name, sequence_list, allow_pickle = True)  #saves a numpy array with every foldercount
    print(finished_vector.shape)
    return(finished_vector,sequence_list)
    
    
            

if __name__ == '__main__':
    valid_data,sequence_list = get_data_from_path(path = "/home/users/username/Masterarbeit/Pronostia_LSTM/Femto_Bearing/Full_Test_set",name = "original.npy")
    
    #valid_data = np.load("/home/users/username/Masterarbeit/Pronostia_LSTM/test2.npy")
    #valid_data = np.transpose(valid_data)
    #print(valid_data)
    
    np.save("test.npy",valid_data)
    valid_folders_count = 0
    for _, dirnames,filenames in os.walk("/home/users/username/Masterarbeit/Pronostia_LSTM/Femto_Bearing/Valid_set"):
        valid_folders_count += len(dirnames)
        
    valid_inp = valid_data[:,:-1]
    valid_label = valid_data[:,1:]
    print(valid_inp.shape)
    valid_inp = valid_inp.reshape(valid_folders_count,-1,valid_inp.shape[1])
    valid_label = valid_label.reshape(valid_folders_count,-1,valid_label.shape[1])
    #print(valid_inp)
    Trainset = TimeSeriesLoader(valid_inp,valid_label)
    Trainloader = DataLoader(Trainset, 9, shuffle = False, drop_last=True, num_workers=0)
    
    
    for inp, label in Trainloader:
        print(inp)
        
        


   