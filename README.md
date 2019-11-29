# LSTM-Lifetime-Estimation
This LSTM is used to predict the rest useful lifetime of ball-bearings. Its programmed with Pytorch and uses the PRONOSTIA Dataset.

To start the program open a terminal and start the "lstm_main.py"
The program checks if there are extracted features saved. (training_data_multi.npy, valid_data_multi.npy)

If there is are no features saved, the user will be asked to give a directory where the Dataset can be found (example: ./Femto_Bearing/Learning_set) 
"lstm_main.py" will call "feautres.py" to generate features. 

"model.py" creates a LSTM with the given hyperparameters. (These need to be changed in the script if you want to use different Hyperparameters)
"training.py" trains the created LSTM and saves the model state dict as "Checkpoint.pth"
A plot of the training loss and validation loss will be created after training, to check the performance of the training

"predict.py" makes a prediction, based on the given input sequence. The length of the prediction, which feature and which bearing wants to be observed can be changed in the "lstm_main.py" script.

"process.py" post processes the "input_results.py" that was created by the "predict.py" class

All named classes will be called when "lstm_main.py" is started

Used libararies:
pandas, Numpy, Scipy, matplotlib, sklearn, pytorch

*Note
Use the Full_Test_set not the validation set, otherwise the post processing won't work
The program sorts the directories by date of change not by name!

*Fix 1.11.2019 Due to the csv format of Bearing 1.4 it wasn't possible to extract features. This has been fixed now
