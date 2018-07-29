#!/usr/bin/env python3
"""
Created on 2018-07-29

AUTHORS: Enrico Sanna - Unversita' degli Studi Guglielmo Marconi - Rome (IT)

PURPOSE: Script that create a RNN LSTM Model to predict values of PYSIONET / CinC Challenge 2018

"""


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import physionetchallenge2018_lib as phyc

import os
import glob
import shutil

from keras.callbacks import EarlyStopping, ModelCheckpoint,CSVLogger,TensorBoard
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import Callback,warnings
import my_logger as L

"""
************************************************************************************
MODEL PARAMETERS
************************************************************************************
Fs=200               frequenza di campionamento segnali
p_WINDOW_SIZE=3*Fs   dimensione della serie storica passata al modello
p_INPUT_FEAT=13      numero di segnali in input
p_OUTPUT_CLASS=3     # 1,0,-1 (total of 3) - numero di classi contenute nel tracciato di target y (arousals)
p_BATCH_SIZE=1000    numero di campioni per volta passati al modello    
p_EPOCHS=75          epoche, numero di volte che lo stesso report viene fatto ripassare.
p_DATASET_DIR        directory del dataset
p_MODEL_FILE         file dove salvo i pesi del modello
p_LOG_FILE           log generale sintetico con AUPRC e AUROC
p_KERAS_LOG_FILE     log testuale in CSV
p_ENTRY_ZIP_FILE     nome del file finale prodotto
p_TENSORBOARD_LOGDIR directory di log per Tensorboard
************************************************************************************
"""
Fs=200
p_WINDOW_SIZE=3*Fs
p_INPUT_FEAT=13
p_OUTPUT_CLASS=3  # 1,0,-1 (total of 3)
p_BATCH_SIZE=1000
p_EPOCHS=75
p_MODEL_NAME="LSTM"
p_MODEL_FILE=str(p_MODEL_NAME)+"_input_"+str(p_INPUT_FEAT)+"_w"+str(p_WINDOW_SIZE)+"_b"+str(p_BATCH_SIZE)+"_e"+str(p_EPOCHS)+".hdf5"
p_LOG_FILE=str(p_MODEL_NAME)+"_input_"+str(p_INPUT_FEAT)+"_w"+str(p_WINDOW_SIZE)+"_b"+str(p_BATCH_SIZE)+"_e"+str(p_EPOCHS)+".log"
p_KERAS_LOG_FILE=str(p_MODEL_NAME)+"_input_"+str(p_INPUT_FEAT)+"_w"+str(p_WINDOW_SIZE)+"_b"+str(p_BATCH_SIZE)+"_e"+str(p_EPOCHS)+"_Keras.log"
p_ENTRY_ZIP_FILE=str(p_MODEL_NAME)+"_input_"+str(p_INPUT_FEAT)+"_w"+str(p_WINDOW_SIZE)+"_b"+str(p_BATCH_SIZE)+"_e"+str(p_EPOCHS)+"_entry.zip"
p_TENSORBOARD_LOGDIR=str(p_MODEL_NAME)+"_input_"+str(p_INPUT_FEAT)+"_w"+str(p_WINDOW_SIZE)+"_b"+str(p_BATCH_SIZE)+"_e"+str(p_EPOCHS)

"""
Inzializzo il modello e stampo i parametri
"""
def init():

    # Create the 'models' subdirectory and delete any existing model files
    try:
        os.mkdir('models')
    except OSError:
        pass
    # Create the 'tensorboard' subdirectory
    try:
        os.mkdir('tensorboard')
    except OSError:
        pass

    for f in glob.glob('models/LSTM_*.hdf5'):
        os.remove(f)

    for f in glob.glob('tensorboard/*'):
        shutil.rmtree(f, ignore_errors=True)

    stringInit = "";
    stringInit += str("\r\n*************************** init ***********************************")
    stringInit += str("\r\nFs (frequenza di campionamento segnali): " + str(Fs))
    stringInit += str(
        "\r\np_WINDOW_SIZE=x*Fs   dimensione della serie storica passata al modello: " + str(p_WINDOW_SIZE))
    stringInit += str("\r\np_INPUT_FEAT=13      numero di segnali in input: " + str(p_INPUT_FEAT))
    stringInit += str(
        "\r\np_OUTPUT_CLASS=3     # 1,0,-1 (total of 3) - numero di classi contenute nel tracciato di target y (arousals):  " + str(
            p_OUTPUT_CLASS))
    stringInit += str("\r\np_BATCH_SIZE=1000    numero di campioni per volta passati al modello " + str(p_BATCH_SIZE))
    stringInit += str(
        "\r\np_EPOCHS=75          epoche, numero di volte che lo stesso report viene fatto ripassare: " + str(p_EPOCHS))
    stringInit += str("\r\np_MODEL_FILE - file dove salvo i pesi del modello:" + str(p_MODEL_FILE))
    stringInit += str("\r\np_DATASET_DIR - directory del dataset:" + str(phyc.p_DATASET_DIR))
    stringInit += str("\r\np_LOG_FILE - log testuale in CSV:" + str(p_LOG_FILE))
    stringInit += str("\r\np_KERAS_LOG_FILE - log testuale in CSV:" + str(p_KERAS_LOG_FILE))
    stringInit += str("\r\np_TENSORBOARD_LOGDIR - directory di log per Tensorboard:" + str(p_TENSORBOARD_LOGDIR))
    stringInit += str("\r\n********************************************************************")
    L.log_info(stringInit)


def finish():
    pass

def loaddata(record_name):
    L.log_info("Loading record: " + str(record_name))
    header_file = record_name + '.hea'
    signal_file = record_name + '.mat'
    arousal_file = record_name + '-arousal.mat'
    # Get the signal names from the header file
    signal_names, Fs, n_samples = phyc.import_signal_names(header_file)
    signal_names = list(np.append(signal_names, 'arousals'))
    this_data = phyc.get_subject_data(arousal_file, signal_file, signal_names)
    SaO2 = this_data.get(['SaO2']).values
    arousals = this_data.get(['arousals']).values
    recordLength = SaO2.size
    #print(this_data)
    #print(this_data.values)
    signals = this_data[
        ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'E1-M2', 'Chin1-Chin2', 'ABD', 'CHEST', 'AIRFLOW',
         'SaO2', 'ECG']]
    signals = signals.values
    arousals=arousals.astype(np.int32)
    return signals,arousals, recordLength

"""
 faccio una scalatura dei segnali di input del deataset di train e test nell'intervallo -1, 1
"""
def scale(dataset):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(dataset)
    # transform train
    dataset = dataset.reshape(dataset.shape[0], dataset.shape[1])
    train_scaled = scaler.transform(dataset)
    # transform test
    return train_scaled

"""
prepare imput for LSTM in 3D vector format
Genero array di lunghezza n contenenti il valore corrente del segnale e gli n-1 precdenti
"""
def strided_axis0_backward(inArr, L = 2):
    # INPUTS :
    # a : Input array
    # L : Length along rows to be cut to create per subarray
    # Append the last row to the start. It just helps in keeping a view output.
    a = np.vstack(( inArr[-L+1:], inArr ))
    # Store shape and strides info
    m,n = a.shape
    s0,s1 = a.strides
    # Length of 3D output array along its axis=0
    nd0 = m - L + 1
    strided = np.lib.stride_tricks.as_strided
    return strided(a[L-1:], shape=(nd0,L,n), strides=(s0,-s0,s1))


###################################################################
### Callback method for reducing learning rate during training  ###
###################################################################
class AdvancedLearnignRateScheduler(Callback):
    '''
   # Arguments
       monitor: quantity to be monitored.
       patience: number of epochs with no improvement
           after which training will be stopped.
       verbose: verbosity mode.
       mode: one of {auto, min, max}. In 'min' mode,
           training will stop when the quantity
           monitored has stopped decreasing; in 'max'
           mode it will stop when the quantity
           monitored has stopped increasing.
   '''

    def __init__(self, monitor='val_loss', patience=0, verbose=0, mode='auto', decayRatio=0.1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.decayRatio = decayRatio

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('Mode %s is unknown, '
                          'fallback to auto mode.'
                          % (self.mode), RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        current_lr = K.get_value(self.model.optimizer.lr)
        print("\nLearning rate:", current_lr)
        if current is None:
            warnings.warn('AdvancedLearnignRateScheduler'
                          ' requires %s available!' %
                          (self.monitor), RuntimeWarning)

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('\nEpoch %05d: reducing learning rate' % (epoch))
                    assert hasattr(self.model.optimizer, 'lr'), \
                        'Optimizer must have a "lr" attribute.'
                    current_lr = K.get_value(self.model.optimizer.lr)
                    new_lr = current_lr * self.decayRatio
                    K.set_value(self.model.optimizer.lr, new_lr)
                    self.wait = 0
            self.wait += 1


def LSTM_model():
    print('Build model LSTM...')
    model = Sequential()
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, input_shape=(None, p_INPUT_FEAT)))
    model.add(Dense(p_OUTPUT_CLASS, activation='relu'))  #better perform relu or hard_sigmoid

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    try:
      model.load_weights('models/'+str(p_MODEL_FILE))
    except:
        print("non ho trovato i pesi. parto a  zero")
    return model



def preprocess_record(record_name):
    signals,arousals, recordLength = loaddata(record_name)

    # Ignore records that do not contain any arousals
    if 1 not in arousals:
        L.log_info('no arousals found in %s\n' % record_name)
        return

    #scalo i segnali di input nell'intervallo da -1 a 1
    signals=scale(signals)
    #x=signals.reshape(1, recordLength, 13)
    x=strided_axis0_backward(signals,p_WINDOW_SIZE)
    y=arousals
    print("x shape: "+ str(x.shape))
    print("y shape: "+ str(y.shape))

    # Convert y2 to dummy variables
    y2 = np.zeros((y.shape[0], p_OUTPUT_CLASS), dtype=np.float32)

    index=0;
    for yi in np.nditer(y):
        if yi == 0:
            y2[index, 0] = 1  # print("messo a zero")
        if yi == 1:
            y2[index, 1] = 1
        if yi == -1:
            y2[index, 2] = 1  # print("messo a  - uno")
        index = index + 1

    #y2[np.arange(y.shape[0]), y] = 1.0
    print("y2 shape: "+ str(y2.shape))
   # print("x:  "+str(x))
    #print("y:  "+str(y))
    #print("y2:  "+str(y2))

    y2sum= y2.sum(axis=0)

    print("y2 sum: "+ str(y2sum))

    callbacks = [
        # Early stopping definition
        EarlyStopping(monitor='val_loss', patience=3, verbose=1),
        # Decrease learning rate by 0.1 factor
        AdvancedLearnignRateScheduler(monitor='val_loss', patience=1, verbose=1, mode='auto', decayRatio=0.1),
        # Saving best model
        ModelCheckpoint('models/'+str(p_MODEL_FILE), monitor='val_loss', save_best_only=False,
                        verbose=1),
        CSVLogger('logs/'+p_KERAS_LOG_FILE, separator=',', append=False),
        TensorBoard(log_dir='tensorboard/'+str(p_TENSORBOARD_LOGDIR), histogram_freq=0, batch_size=32, write_graph=True,
                                    write_grads=False, write_images=False, embeddings_freq=0,
                                    embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)
    ]

    model =LSTM_model()

    # split train and validation sets

    idxval_start = np.random.randint(np.trunc(recordLength/2), size=1)[0]
    idxval_size = np.random.randint(np.trunc(recordLength / 2), size=1)[0]
    #print(idxval_start)
    #print(idxval_size)
    Xval = x[idxval_start:idxval_start+idxval_size, :, :]
    Yval = y2[idxval_start:idxval_start + idxval_size, :]
    print('Train...')
    model.fit(x,y2, validation_data=(Xval,Yval),epochs=p_EPOCHS,batch_size=p_BATCH_SIZE, callbacks=callbacks) #200
    pred = model.predict(x)
    predict_classes = np.argmax(pred,axis=1)
    predict_classes[predict_classes == 2] = -1

    #print("Predicted classes: {}",predict_classes)
    #print("Expected classes: {}",y)
    print("Predicted sum: {}",predict_classes.sum(axis=0))
    print("Expected sum: {}",y.sum(axis=0))

    try:
      model.save_weights('models/'+str(p_MODEL_FILE))
    except:
        L.log_info("non ho trovato i pesi. parto a  zero")

def classify_record(record_name):
    signals, arousals, recordLength = loaddata(record_name)
    # scalo i segnali di input nell'intervallo da -1 a 1
    signals = scale(signals)
    # x=signals.reshape(1, recordLength, 13)
    x = strided_axis0_backward(signals, p_WINDOW_SIZE)
    model = LSTM_model()
    # Need to add dimension for training
    predictions = model.predict(x, batch_size=p_BATCH_SIZE)
    predict_classes = np.argmax(predictions, axis=1)
    #porto a 0 anziché -1 perché da errore in fase di scoring
    predict_classes[predict_classes == 2] = 0
    #print("predict_classes"+str(predict_classes))
    #print("pred" + str(pred))
    #return predict_classes,
    pred_arousal_probabilities=predictions[:,1].clip(min=0)
    return  predict_classes,pred_arousal_probabilities
