#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 01:25:57 2020

data process from scratch

@author: ericwu
"""

import numpy as np
import pickle
from tqdm import tqdm
import os
from scipy.io import loadmat
import pandas as pd
from biosppy.signals import ecg as ecgprocess
import matplotlib.pyplot as plt
import cv2
import time
from scipy.signal import spectrogram
import gc

gc.enable()

def SaveAsPickle(varables,file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(varables, f)
def LoadPickle(file_name):
    file = pickle.load(open(file_name, 'rb'))
    return file
    


def ImportFile(path):
    
    file_list = []
      
    #root, directories, files
    for r, d, f in os.walk(path):        
        for file in f:
            if '.mat' in file:
                         
                file_dir = os.path.join(r, file)
                file_list.append(file_dir)
                
    file_list = sorted(file_list) #sorted list by file's name (eg. A00001 to A8528)
    
    signals = []
    for file in tqdm(file_list):
        sig = list(loadmat(file).values())[0][0]/1000
        signals.append(sig)
     
    #import reference 
    refer_path = os.path.join(path, 'REFERENCE.csv')
    reference = np.array(pd.read_csv(refer_path, header = None))
    label = reference[:,1]
    label[label == 'N'] = 0  #Normal
    label[label == 'A'] = 1  #Afib
    label[label == 'O'] = 2  #Other
    label[label == '~'] = 3  #Noise
    
    dataset = list(zip(label, signals))
           
    return dataset

def WindowSelection(signals, win_size= 3000, method= 'center', StartPoint = None):
    
    print('select window...')
    
    if method == 'center':
        Signals = []
        
        for sig in tqdm(signals):
            
            sig_len = len(sig)
            if sig_len < win_size:
                pad_num = win_size - sig_len
                pad_left = int(np.ceil(pad_num/2))
                pad_right = int(np.floor(pad_num/2))
                select_signal_win = np.pad(sig, (pad_left, pad_right), mode = 'constant')
                Signals.append(select_signal_win)
            else:
                
                start_point = int(np.round((sig_len - win_size)/2))
                end_point = start_point + win_size
                select_signal_win = sig[start_point:end_point]
                Signals.append(select_signal_win)
        
        return Signals
            
    elif method == 'random':
        Signals = []
        
        for sig in tqdm(signals):
            
            sig_len = len(sig)
            if sig_len < win_size:
                pad_num = win_size - sig_len
                pad_left = int(np.ceil(pad_num/2))
                pad_right = int(np.floor(pad_num/2))
                select_signal_win = np.pad(sig, (pad_left, pad_right), mode = 'constant')
                Signals.append(select_signal_win)
            else:
                max_start_point = sig_len - win_size
                start_point = np.random.randint(0, max_start_point)
                end_point = start_point + win_size
                select_signal_win = sig[start_point:end_point]
        return Signals
    
    elif method == 'fix':
        Signals = []
        
        for sig in tqdm(signals):
            
            sig_len = len(sig)
            if sig_len < win_size:
                pad_num = win_size - sig_len
                pad_left = int(np.ceil(pad_num/2))
                pad_right = int(np.floor(pad_num/2))
                select_signal_win = np.pad(sig, (pad_left, pad_right), mode = 'constant')
                Signals.append(select_signal_win)
            else:
                start_point = StartPoint
                end_point = start_point + win_size
                select_signal_win = sig[start_point:end_point]
                Signals.append(select_signal_win)
        
        return Signals
    
    else:
        print('Error: Please select method.')

def FeatureExtraction(dataset):
    Ts            = [] #Signal time axis reference (seconds).
    Filtered_ecg  = [] #Filtered ECG signal.
    Rpeaks        = [] #R-peak location indices.
    Templates_ts  = [] #Templates time axis reference (seconds).
    Templates     = [] #Extracted heartbeat templates.
    Heart_rate_ts = [] #Heart rate time axis reference (seconds).
    Heart_rate    = [] #Instantaneous heart rate (bpm).
    Label         = []
    for lb, sig in tqdm(dataset):
        ts, filt_ecg, rp, temp_ts, temp, hr_ts, hr = ecgprocess.ecg(sig, 300, False)
        
        Ts.append(ts)
        Filtered_ecg.append(filt_ecg)
        Rpeaks.append(rp)
        Templates_ts.append(temp_ts)
        Templates.append(temp)
        Heart_rate_ts.append(hr_ts)
        Heart_rate.append(hr)
        Label.append(lb)
        
    return Ts, Filtered_ecg, Rpeaks, Templates_ts, Templates, Heart_rate_ts, Heart_rate, Label 

#3000 iterations at one time, to avoid out of memory 
def PrepareTemplates(Templates, Templates_ts, save_path, start, end): #add start, end to avoid out of memory
    
    #check data match
    if len(Templates) != len(Templates_ts):
        raise ValueError
        
    for i in tqdm(range(start, end)):
        
        if i > len(Templates) - 1:
            #creat image file list
            file_list = []
            for j in range(len(Templates)):
                file = str(j)+'.png'
                file_list.append(file)    
            SaveAsPickle(file_list, os.path.join(save_path,'file_list.pk1'))
            break
        
        else:
            plt.plot(Templates_ts[i], Templates[i].T, 'm' ,alpha = 0.7)
            plt.axis('off')   #don't show axis
            plt.savefig(os.path.join(save_path, str(i)))
            plt.clf()
            plt.close('all')
            
            
            
            
                    
                                                
def Image2Array(path, file_list, label, img_size, negative =False):
    
    IMG_array = []
    
    for img in tqdm(file_list):
        
        img_path = os.path.join(path, img)
        
        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array, (img_size, img_size))
        
        if negative == True:
            img_array = abs(255-img_array)
        
        IMG_array.append(img_array)
        
        dataset = list(zip(label,IMG_array))
        
    return dataset

def PrepareArgumentation(dataset):
    image_array = []
    label = []
    for lb, img in tqdm(dataset):
        label.append(lb)
        image_array(img)
        
    return image_array, label


def FindMedianAmp(templates):
    Amplitude = []
    for temp in tqdm(templates):
        peak = []
        for amp in temp:
            peak.append(max(amp))
        med_val = np.percentile(peak, 50)
        
        select = (abs(peak - med_val)).argmin()
        
        Amplitude.append(temp[select])
        
    return Amplitude


def PrepareFeatures(train_path = './training2017', sample_path = './sample2017/validation'):
    
    #training dataset feature extraction
    training = ImportFile(train_path)
    if os.path.isdir('./feature') == False:
        os.mkdir('./feature') #creat folder to save features
    tn_Ts, tn_Filtered_ecg, tn_Rpeaks, tn_Templates_ts, tn_Templates, tn_Heart_rate_ts, tn_Heart_rate, tn_Label = FeatureExtraction(training)
    
    SaveAsPickle(tn_Ts, './feature/train_ts.pk1')
    SaveAsPickle(tn_Filtered_ecg, './feature/train_filtered_ecg.pk1')
    SaveAsPickle(tn_Rpeaks, './feature/train_Rpeak.pk1')
    SaveAsPickle(tn_Templates_ts, './feature/train_templates_ts.pk1')
    SaveAsPickle(tn_Templates, './feature/train_templates.pk1')
    SaveAsPickle(tn_Heart_rate_ts, './feature/train_HeartRate_ts.pk1')
    SaveAsPickle(tn_Heart_rate, './feature/train_HeartRate.pk1')
    SaveAsPickle(tn_Label, './feature/train_label.pk1')
    
    del tn_Ts, tn_Filtered_ecg, tn_Rpeaks, tn_Heart_rate_ts, tn_Heart_rate, tn_Label
    
    
    #validation dataset feature extraction
    sample = ImportFile(sample_path)
    
    tt_Ts, tt_Filtered_ecg, tt_Rpeaks, tt_Templates_ts, tt_Templates, tt_Heart_rate_ts, tt_Heart_rate, tt_Label = FeatureExtraction(sample)
    
    SaveAsPickle(tt_Ts, './feature/test_ts.pk1')
    SaveAsPickle(tt_Filtered_ecg, './feature/test_filtered_ecg.pk1')
    SaveAsPickle(tt_Rpeaks, './feature/test_Rpeak.pk1')
    SaveAsPickle(tt_Templates_ts, './feature/test_templates_ts.pk1')
    SaveAsPickle(tt_Templates, './feature/test_templates.pk1')
    SaveAsPickle(tt_Heart_rate_ts, './feature/test_HeartRate_ts.pk1')
    SaveAsPickle(tt_Heart_rate, './feature/test_HeartRate.pk1')
    SaveAsPickle(tt_Label, './feature/test_label.pk1')
    
    del tt_Ts, tt_Filtered_ecg, tt_Rpeaks, tt_Heart_rate_ts, tt_Heart_rate, tt_Label
    
    return tn_Templates, tn_Templates_ts, tt_Templates, tt_Templates_ts

def SaveTemplates(tn_Templates, tn_Templates_ts, tt_Templates, tt_Templates_ts):
    if os.path.isdir('./ECG_templates') == False:
        os.mkdir('./ECG_templates') #creat folder
    if os.path.isdir('./ECG_templates/train') == False:
        os.mkdir('./ECG_templates/train')
    if os.path.isdir('./ECG_templates/test') == False:
        os.mkdir('./ECG_templates/test')
    
    save_path_1 = './ECG_templates/train'  
    PrepareTemplates(tn_Templates, tn_Templates_ts, save_path_1, 0, 3000 )
    time.sleep(10)
    PrepareTemplates(tn_Templates, tn_Templates_ts, save_path_1, 3000, 6000)
    time.sleep(10)
    PrepareTemplates(tn_Templates, tn_Templates_ts, save_path_1, 6000, 9000)
    time.sleep(10)
    
    save_path_2 = './ECG_templates/test'   
    PrepareTemplates(tt_Templates, tt_Templates_ts, save_path_2, 0, 1000)
    
def PrepareSpectrogram(tn_Filtered_ecg, tt_Filtered_ecg):
    
    tn_cut_sig = WindowSelection(tn_Filtered_ecg, win_size = 9000, method = 'center')
    tn_t = []
    tn_f = []
    tn_sxx = []
    for sig in tqdm(tn_cut_sig):
        
        f, t, sxx = spectrogram(sig, 300, nperseg = 64, noverlap = 0.5)
        sxx = np.log(sxx)
        tn_f.append(f)
        tn_t.append(t)
        tn_sxx.append(sxx)
    
    
    tt_cut_sig = WindowSelection(tt_Filtered_ecg, win_size = 9000, method = 'center')
    tt_t = []  
    tt_f = []
    tt_sxx = []
    for sig in tqdm(tt_cut_sig):
        
        f, t, sxx = spectrogram(sig, 300, nperseg = 64, noverlap = 0.5)
        sxx = np.log(sxx)
        tt_f.append(f)
        tt_t.append(t)
        tt_sxx.append(sxx)
    
    return tn_f, tn_t, tn_sxx, tt_f, tt_t, tt_sxx
    
def Signal2Spectrogram(spec_sxx, spec_f, spec_t, save_path, start, end):
    
    
    for i in tqdm(range(start, end)):
        
        if i > len(spec_sxx) - 1:
            file_list = []
            for j in range(len(spec_sxx)):
                file = str(j) + '.png'
                file_list.append(file)
            SaveAsPickle(file_list, os.path.join(save_path,'file_list.pk1' ))
            break
                
            
        else:
            plt.pcolormesh(spec_t[i], spec_f[i], spec_sxx[i])
            plt.axis('off')
            plt.savefig(os.path.join(save_path, str(i)), facecolor = 'xkcd:black')
            plt.clf()
            plt.close('all')
        
def SaveSpectrogram(tn_f, tn_t, tn_sxx, tt_f, tt_t, tt_sxx):
    
    if len(tn_f) != len(tn_sxx) or len(tt_f) != len(tt_sxx):
        raise ValueError
    
    
    #creat folder
    if os.path.isdir('./ECG_spectrogram') == False:
        os.mkdir('./ECG_spectrogram')
    if os.path.isdir('./ECG_spectrogram/train') == False:
        os.mkdir('./ECG_spectrogram/train')
    if os.path.isdir('./ECG_spectrogram/test') == False:
        os.mkdir('./ECG_spectrogram/test')
    
    #save figure
    save_path_1 = './ECG_spectrogram/train'
    Signal2Spectrogram(tn_sxx, tn_f, tn_t, save_path_1, 0, 3000) #total 8528 records
    time.sleep(10)
    Signal2Spectrogram(tn_sxx, tn_f, tn_t, save_path_1, 3000, 6000)
    time.sleep(10)
    Signal2Spectrogram(tn_sxx, tn_f, tn_t, save_path_1, 6000, 9000)
    time.sleep(10)
    
    save_path_2 = './ECG_spectrogram/test'
    Signal2Spectrogram(tt_sxx, tt_f, tt_t, save_path_2, 0, 1000) #total 300 records

def get_MedAmpInput():
     
    #tn_Templates, tn_Templates_ts, tt_Templates, tt_Templates_ts = PrepareFeatures()
    tn_Templates    = LoadPickle('./feature/train_templates.pk1')
    
    tt_Templates    = LoadPickle('./feature/test_templates.pk1')
    
    train_sig = FindMedianAmp(tn_Templates)
    test_sig  = FindMedianAmp(tt_Templates)
    train_lb  = LoadPickle('./feature/train_label.pk1')
    test_lb   = LoadPickle('./feature/test_label.pk1')
    
    train  = list(zip(train_lb, train_sig))
    test   = list(zip(test_lb, test_sig))
    SaveAsPickle(train, 'train_med_amp.pk1')
    SaveAsPickle(test, 'test_med_amp.pk1')

def get_TempInput():
        
    #tn_Templates, tn_Templates_ts, tt_Templates, tt_Templates_ts = PrepareFeatures()
    tn_Templates    = LoadPickle('./feature/train_templates.pk1')
    tn_Templates_ts = LoadPickle('./feature/train_templates_ts.pk1')
    tt_Templates    = LoadPickle('./feature/test_templates.pk1')
    tt_Templates_ts = LoadPickle('./feature/test_templates_ts.pk1')
    SaveTemplates(tn_Templates, tn_Templates_ts, tt_Templates, tt_Templates_ts)
    file_list_1 = LoadPickle('./ECG_templates/train/file_list.pk1')
    train_label = LoadPickle('./feature/train_label.pk1')
    train_img = Image2Array(path = './ECG_templates/train',
                            file_list = file_list_1,
                            img_size = 64,
                            label = train_label,
                            negative = True)
    
    file_list_2 = LoadPickle('./ECG_templates/test/file_list.pk1')
    test_label = LoadPickle('./feature/test_label.pk1')
    test_img = Image2Array(path = './ECG_templates/test',
                           file_list = file_list_2,
                           img_size = 64,
                           label = test_label,
                           negative = True)
    
    
    SaveAsPickle(train_img, 'train_temp_input.pk1')
    SaveAsPickle(test_img, 'test_temp_input.pk1')
    
    
def get_SpecgInput():
    
    #_, _, _, _ = PrepareFeatures()
    
    tn_Filtered_ecg = LoadPickle('./feature/train_filtered_ecg.pk1')
    tt_Filtered_ecg = LoadPickle('./feature/test_filtered_ecg.pk1')
    
    tn_f, tn_t, tn_sxx, tt_f, tt_t, tt_sxx = PrepareSpectrogram(tn_Filtered_ecg, tt_Filtered_ecg)
    
    SaveSpectrogram(tn_f, tn_t, tn_sxx, tt_f, tt_t, tt_sxx)
    
    file_list_1 = LoadPickle('./ECG_spectrogram/train/file_list.pk1')
    train_label = LoadPickle('./feature/train_label.pk1')
    
    train_img = Image2Array(path = './ECG_spectrogram/train',
                            file_list = file_list_1,
                            img_size = 100,
                            label = train_label,
                            negative = False)
    
    file_list_2 = LoadPickle('./ECG_spectrogram/test/file_list.pk1')
    test_label = LoadPickle('./feature/test_label.pk1')
    
    test_img = Image2Array(path = './ECG_spectrogram/test',
                           file_list = file_list_2,
                           img_size = 100,
                           label = test_label,
                           negative = False)
    
    SaveAsPickle(train_img, 'train_specg_input.pk1')
    SaveAsPickle(test_img, 'test_specg_input.pk1')
    
    

    
if __name__ == "__main__":
    _, _, _, _ = PrepareFeatures()
    get_MedAmpInput()
    get_TempInput()
    get_SpecgInput()

    
    


    
    
        
      
        
    
    
    
    

        
    

    

    


'''
test

_, _, _, _ = PrepareFeatures()
get_MedAmpInput()
get_TempInput()
get_SpecgInput()
'''