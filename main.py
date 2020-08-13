import cnn_model as cnn
import Eval_model
import pickle
import numpy as np

def load_data(file_path):
    file = pickle.load(open(file_path, 'rb'))
    x_data = []
    y_data = []
    for y, x in file:
        x_data.append(x)
        y_data.append(x)

    return np.array(x_data).astype('float32'), np.array(y_data).astype('int8')


if __name__ == "__main__":
    
    #set file path
    train_med_path = './train_med_amp.pk1'
    test_med_path = './test_med_amp.pk1'

    train_temp_path = './train_temp_input.pk1'
    test_temp_path = './test_temp_input.pk1'

    train_specg_path = './train_specg_input.pk1'
    test_specg_path = './test_specg_input.pk1'
    
    # prepare median wave input
    x_med, y_med = load_data(train_med_path)
    x_med_test, y_med_test = load_data(test_med_path)
    x_med, y_med, x_med_test, y_med_test, y_med_true = cnn.InputPreprocess(x_med, y_med, x_med_test, y_med_test)
    # prepare heartbeat template input
    x_temp, y_temp = load_data(train_temp_path)
    x_temp_test, y_temp_test = load_data(test_temp_path)
    x_temp, y_temp, x_temp_test, y_temp_test, y_temp_true = cnn.InputPreprocess(x_temp, y_temp, x_temp_test, y_temp_test, model_type='2d') 
    # prepare spectrogram input 
    x_specg, y_specg = load_data(train_specg_path)
    x_specg_test, y_specg_test = load_data(test_specg_path)
    x_specg, y_specg, x_specg_test, y_specg_test, y_specg_true = cnn.InputPreprocess(x_specg, y_specg, x_specg_test, y_specg_test, model_type='2d')

    #training data
    md_med, his_med = cnn.get_1DCNN(x_med, y_med, x_med_test, y_med_test, name='med') #1D-CNN
    md_temp, his_temp = cnn.get_2DCNN(x_temp, y_temp, x_temp_test, y_temp_test, name='temp') #2D-CNN
    md_specg, his_specg = cnn.get_2DCNN(x_specg, y_specg, x_specg_test, y_specg_test, name='specg') #2D-CNN

    

