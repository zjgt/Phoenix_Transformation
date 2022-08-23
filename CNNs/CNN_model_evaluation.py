import os
import keras
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
from scikitplot.metrics import plot_confusion_matrix, plot_roc
import tensorflow as tf
from sklearn import metrics
from keras import backend as K
import glob
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import pickle
import time
from sklearn.metrics import confusion_matrix
import image_dataset_loader

src_dir = 'G5758/'
data_folder = 'G5758/test'

modelist = [f for f in os.listdir('G5758/') if f.endswith('.h5')]

(x_test, y_test), = image_dataset_loader.load(data_folder, ['test'])
validation_data = x_test, y_test

for file in modelist:
    modelfile = src_dir + file
    model = tf.keras.models.load_model(modelfile)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    plt.clf()
    fig, ax = plt.subplots(figsize=(5, 5))
    y_pred = model.predict(x_test)
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    plt.plot(fpr, tpr, label="AUC=" + str(auc)[:6], color='red')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='lower right')
    fig.savefig(os.path.join(src_dir, f'roc_curve_epoch_{file}.png'))

    plt.clf()
    fig, ax = plt.subplots(figsize=(5, 5))
    y_pred = (model.predict(x_test) >= 0.5).astype(bool)
    y_true = validation_data[1]
    plot_confusion_matrix(y_true, y_pred, ax=ax)
    fig.savefig(os.path.join(src_dir, f'confusion_matrix_{file}.png'))
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    precision = TP / (TP + FP + K.epsilon())
    recall = TP / (TP + FN + K.epsilon())
    F1_score = 2 * precision * recall / (precision + recall + K.epsilon())
    print("Precision score is:" + str(precision)[:6])
    print("Recall score is:" + str(recall)[:6])
    print("F1_score is:" + str(F1_score)[:6])

    plt.clf()




