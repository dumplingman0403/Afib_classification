from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Activation
from tensorflow.keras.layers import Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard
import numpy as np
import os
from datetime import datetime
logdir = './logs'

def get_1DCNN(x_train, y_train, x_test, y_test, name, epochs = 50):

    model = Sequential()
    model.add(Conv1D(16, 3, activation= 'relu', \
        input_shape = (x_train.shape[1], 1)))
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
    now = datetime.now()
    now = now.strftime("%H:%M:%S")
    callback = [EarlyStopping(monitor = 'val_loss', patience = 8),
                ModelCheckpoint(filepath ='1DCNN_best_model.h5', monitor = 'val_loss', save_best_only = True),
                TensorBoard(log_dir='logs/{}{}'.format(now, name), histogram_freq=1)]

    history = model.fit(x_train, y_train, 
                        batch_size = 32, 
                        epochs = epochs, 
                        callbacks = callback, 
                        validation_data = (x_test, y_test))
    
    return model, history





def get_2DCNN(x_train, y_train, x_test, y_test, name, epochs =50):
    nClass = 4
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = x_train.shape[1:]))
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(nClass, activation = 'softmax' ))

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                  metrics = ['accuracy'])
    now = datetime.now()
    now = now.strftime("%H:%M:%S")
    callbacks = [EarlyStopping(monitor='val_loss', patience = 20), 
                 ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True),
                 TensorBoard(log_dir= 'logs/{}{}'.format(now, name), histogram_freq=1)
                 ]
    history = model.fit(x_train, y_train, batch_size = 32, epochs = epochs,
                        callbacks = callbacks, validation_data = (x_test, y_test) )
    
    #model.load_weights('best_model.h5')
    model.summary()
    return model, history

def InputPreprocess(x_train, y_train, x_test, y_test, model_type = '1d'):
    if model_type == '1d':

        #reshape x
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] , 1))
        x_test  = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
        y_true = y_test
        #reshape y 
        y_test = to_categorical(y_test, 4, dtype = 'int8')
        y_train = to_categorical(y_train, 4, dtype = 'int8')
    
        return x_train, y_train, x_test, y_test, y_true

    elif model_type == '2d':
        #reshape
        x_train /= 255
        x_test  /= 255
    
        x_train = np.reshape(x_train, (-1, IMG_SIZE, IMG_SIZE, 1))
        x_test = np.reshape(x_test, (-1, IMG_SIZE, IMG_SIZE, 1))

    
        y_true = y_test
        #to categorical label

        y_train = to_categorical(y_train, num_classes = 4, dtype = 'int8')
        y_test = to_categorical(y_test, num_classes = 4, dtype = 'int8' )
    
        return x_train, y_train, x_test, y_test, y_true
    else:
        raise ValueError('Select model type 1d or 2d.')