#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 04:02:30 2020

1D CNN model new

@author: eric
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Activation
from tensorflow.keras.layers import Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard
import Eval_model as eva
import pickle

def get_1DCNN(x_train, y_train, x_test, y_test, epochs = 50):
    
    model = Sequential()
    
    model.add(Conv1D(16, 3, activation = 'relu', input_shape = (x_train.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    
    model.add(Conv1D(32, 3, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    
    model.add(Conv1D(64, 3, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    
    model.add(Flatten())
    model.add(Dropout(0.5))
    
    
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    
    model.add(Dense(4, activation = 'softmax'))
    
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    
    callback = [EarlyStopping(monitor = 'val_loss', patience = 8),
                ModelCheckpoint(filepath ='1DCNN_best_model.h5', monitor = 'val_loss', save_best_only = True)]
    
   #callback = [TensorBoard]
    
    history = model.fit(x_train, y_train, 
                        batch_size = 32, 
                        epochs = epochs, 
                        callbacks = callback, 
                        validation_data = (x_test, y_test))
    
    return history, model

def InputPreprocess(x_train, y_train, x_test, y_test):
    #reshape x
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] , 1))
    x_test  = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    y_true = y_test
    #reshape y 
    y_test = to_categorical(y_test, 4, dtype = 'int8')
    y_train = to_categorical(y_train, 4, dtype = 'int8')
    
    return x_train, y_train, x_test, y_test, y_true


x_train = np.array(pickle.load(open('/home/eric/1D_CNN/still test/tran_med_amp.pk1', 'rb')))
x_test  = np.array(pickle.load(open('/home/eric/1D_CNN/still test/test_med_amp.pk1', 'rb'))) 
y_train = np.array(pickle.load(open('y_train_2.pickle', 'rb'))).astype('int8')
y_test  = np.array(pickle.load(open('y_test_2.pickle', 'rb'))).astype('int8')


x_train, y_train, x_test, y_test, y_true = InputPreprocess(x_train, y_train, x_test, y_test)
history, model = get_1DCNN(x_train, y_train, x_test, y_test)
model.summary()
eva.get_summary(x_test, y_true, model)
eva.Plot_Acc_and_Loss(history, x_test, y_test, model)
    
    
    



    


                  
                  

