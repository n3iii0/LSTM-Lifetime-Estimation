# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def get_current_RUL(bearing):
    RUL_dict = {
      0  : 1802,
      1 : 1139,
      2 : 2302,
      3 : 2302,
      4 : 1502,
      5 : 1202,
      6 : 612,
      7 : 2002,
      8 : 572,
      9 : 172,
      10 :352,
     }
    
    return RUL_dict[bearing]

def get_actual_RUL(bearing):
    RUL_dict = {
      0  : 573,
      1 : 290,
      2 : 161,
      3 : 146,
      4 : 752,
      5 : 753,
      6 : 139,
      7 : 309,
      8 : 129,
      9 : 58,
      10 : 82,
     }
    
    return RUL_dict[bearing]
    
def remove_padding(data,pad):
    data = data[pad:]
    print(data.shape)
    return data

'''finds the zeropoint of input data
if the selected feature is the RUL the zero_point of the curve is equals the RUL
filtering is needed in case the zero-point is at the beginning of the curve, fv = 0 no filter by default'''
def find_zero_point(data, fv):
    zeros = []
    filtering = fv
    for i,x in enumerate(data[filtering:]):
        if data[filtering:][i] <= 0:
            zeros.append(i)
            
    try:
        print("RUL ends at ",zeros[0]+filtering)
        zeros[0]+filtering
    except IndexError: 
        print("RUL hasn't been reached yet")

'''input: Sequence List of all Data, Dataset, Bearing number, Feature thats inspected'''
'''output: A Result Graph, and the RUL Time in das'''
'''The input_result.npy data that has been created by the predict class is beeing loaded and processed
for that the padding is beeing removed and the input sequence and the ending rul is beeing plotted'''

def post_processing(sequence_list, Validset, bearing, feature):
    
    filter_value = 1
    inp, label = Validset[bearing]
    actual_sequence = inp.shape[1]
    RUL_Sequence = get_current_RUL(bearing)+get_actual_RUL(bearing) 
    pad = actual_sequence+1-RUL_Sequence   
    print(pad)
    result = np.load("input_result.npy")
    result = result[:,:,feature].reshape(-1)
    
    result_unpadded = remove_padding(result,pad)
    
    find_zero_point(result_unpadded,filter_value)
    
    
    Full_RUL = label[feature,pad:].reshape(-1)
    input_line = Full_RUL[:get_current_RUL(bearing)-1]
    
    threshold = np.zeros(result.shape[0])
    
    
    plt.plot(Full_RUL, color = 'b', label = "True RUL")
    plt.plot(result_unpadded, color = 'g', label = "Prediction")
    plt.plot(threshold, color = 'r', label = "Threshold")
    plt.plot(input_line, color = '#ffe476', label = "Sequence length")
    
    plt.grid()
    plt.legend()
    plt.xlabel("RUL[das]")
    plt.ylabel("Amplitude")
    plt.savefig("Results.png")
    plt.close()
