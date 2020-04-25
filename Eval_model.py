#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 03:46:36 2020

Get evaluation reslut 

such as accuracy, f1, sensitivity, specificity ... 

@author: ericwu
"""



import numpy as np
import pickle
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import itertools
import matplotlib.pyplot as plt

def EvaluateModel(x_test, y_true, model):
    
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis = -1)
    
    cnf_matrix = confusion_matrix(y_true, y_pred)
    
    TP = np.diag(cnf_matrix)
    FP = cnf_matrix.sum(axis = 0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis = 1) - np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (TP + FP + FN)
    
    Se = TP/(TP + FN)  #sensitivity
    Sp = TN/(TN + FP)  #specificity
    PPV = TP/(TP + FP) #Positive predictive value 
    NPV = TN/(TN + FN) #Negative predictive value
    
    resp_acc = (TP + TN)/(TP + TN + FP +FN) #respective overall accuracy
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred,  average = 'macro')
    
    return acc, Se, Sp, f1, PPV, NPV, cnf_matrix


def plot_confusion_matrix(cm, classes, 
                          normalize = False,
                          title = 'Confusion Matrix',
                          cmap = plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = "center",
                 color = "white" if cm[i, j] >thresh else "black")
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        
        
def get_summary(x_test, y_true, model):
    
    acc, Se, Sp, f1, PPV, NPV, cm = EvaluateModel(x_test, y_true, model)
    classes = ['N', 'A', 'O', '~']
    
    print('Accuracy = %.2f%%' %(acc*100))
    print('F1 score = %.2f' %(f1*100))
    print('Sensitivity: Normal = %.3f, AFib = %.3f, Others = %.3f, Noise = %.3f' %(Se[0], Se[1], Se[2], Se[3]))
    print('Specificity: Normal = %.3f, AFib = %.3f, Others = %.3f, Noise = %.3f' %(Sp[0], Sp[1], Sp[2], Sp[3]))
    print('Positive predictive value: Normal = %.3f, AFib = %.3f, Others = %.3f, Noise = %.3f' %(PPV[0], PPV[1],
                                                                                                 PPV[2], PPV[3]))
    print('Negative predictive value: Normal = %.3f, AFib = %.3f, Others = %.3f, Noise = %.3f' %(NPV[0], NPV[1],
                                                                                                 NPV[2], NPV[3]))
    
    plot_confusion_matrix(cm, classes)
    plt.show()
    plot_confusion_matrix(cm, classes, normalize = True, title = 'Normalized Confusion Matrix' )
    plt.show()
    
def Plot_Acc_and_Loss(history, x_test, y_test, model):
    score = model.evaluate(x_test, y_test, verbose = 0)
    print("Accuracy: %.2f%%" %(score[1]*100))
    #print(history)
    fig1, ax_acc = plt.subplots()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model-Accuracy')
    plt.legend(['Training', "Validation"], loc = 'lower right')
    plt.show()
    
    fig2, ax_loss = plt.subplots()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model-Loss')
    plt.legend(['Training', "Validation"], loc = 'lower right')
    plt.show()
    #target_names = ['0', '1', '2', '3']