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

curr1 = time.time()
#base_dir and subdirectories are subject to change.
base_dir = 'Lung_KL_combined/'
normal = glob.glob(base_dir + 'normal/*.*')
tumor = glob.glob(base_dir + 'tumor/*.*')
data = []
labels = []

for i in normal:
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb',
    target_size= (512,512))
    image=np.array(image)
    data.append(image)
    labels.append(0)

for i in tumor:
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb',
    target_size= (512,512))
    image=np.array(image)
    data.append(image)
    labels.append(1)

data = np.array(data)
labels = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.5)

x_train, x_test = x_train / 255.0, x_test / 255.0
validation_data = x_test, y_test

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(512, 512, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

class PerformanceVisualizationCallback(Callback):
    def __init__(self, model, validation_data, image_dir):
        super().__init__()
        self.model = model
        self.validation_data = validation_data

        os.makedirs(base_dir, exist_ok=True)
        self.image_dir = base_dir

    def on_epoch_end(self, epoch, logs={}):
        curr = time.time()
        #plot and save confusion matrix
        fig, ax = plt.subplots(figsize=(5, 5))
        y_pred = (model.predict(x_test) >= 0.5).astype(bool)
        plot_confusion_matrix(y_test, y_pred, ax=ax)
        fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}_{curr}.png'))

        # plot and save roc curve
        fig, ax = plt.subplots(figsize=(5, 5))
        y_pred = model.predict(x_test)
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
        auc = metrics.roc_auc_score(y_test, y_pred)
        plt.plot(fpr, tpr, label="AUC=" + str(auc)[:6], color='red')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc='lower right')
        fig.savefig(os.path.join(self.image_dir, f'roc_curve_epoch_{epoch}_{curr}.png'))

#use curr1 to ensure only one best model is saved in each run, while use curr to save every epoch's metrics.
chkpt_cb = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(base_dir, f'2layer_cnn_model_{curr1}.h5'),
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=True,
                                              initial_value_threshold=0.25)

performance_viz_cbk = PerformanceVisualizationCallback(
    model=model,
    validation_data=validation_data,
    image_dir=base_dir)

history = model.fit(x=x_train,
                    y=y_train,
                    epochs=25,
                    validation_data=validation_data,
                    callbacks=[performance_viz_cbk, chkpt_cb])


#y_pred = model.predict(x_test)
y_pred = (model.predict(x_test) >= 0.5).astype(bool)
TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
precision = TP/(TP+FP+K.epsilon())
recall = TP/(TP+FN+K.epsilon())
F1_score = 2*precision*recall/(precision+recall+K.epsilon())
print("Precision score is:"+str(precision)[:6])
print("Recall score is:"+str(recall)[:6])
print("F1_score is:"+str(F1_score)[:6])

print(history.history.keys())
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

curr = time.time()
# Get number of epochs
epochs = range(len(acc))
with open(os.path.join(base_dir, f'2layer_binary_adam_TrainHistoryDict_{curr}'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

np.save(os.path.join(base_dir, f'2layer_binary_adam_Training_{curr}.npy'), history.history)

plt.clf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylim(0.5, 1.0)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.savefig(os.path.join(base_dir, f'2layer_binary_adam_accuracy_{curr}.png'))

# Plot training and validation loss per epoch
plt.clf()
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.ylim(0.0, 2.0)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.title('Training and validation loss')
plt.savefig(os.path.join(base_dir, f'2layer_binary_adam_Loss_{curr}.png'))
