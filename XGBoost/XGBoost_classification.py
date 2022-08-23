import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scikitplot.metrics import plot_confusion_matrix, plot_roc
import matplotlib.pyplot as plt
from sklearn import metrics

#load the expression data from a csv file
luadlusc = pd.read_csv("../Revision/xgboost_luad_lusc_expression_truncated.csv", header=0, index_col=0)
dataset = luadlusc

#data transpose so that each column (except column 0) correspond to one gene
dataset = dataset.transpose()
m, n = dataset.shape

#the second column and onward are gene expression data, the first column is disease state
X = dataset.iloc[:, 1:n]
Y = dataset.iloc[:, 0]

#data split and training
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=7)
model = XGBClassifier()
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=False)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
results = model.evals_result()

#loading results and plotting
plt.clf()
epochs = len(results["validation_0"]["error"])
x_axis = range(0, epochs)
fig, ax = plt.subplots(figsize=(5,5))
plot_confusion_matrix(y_test, y_pred, ax=ax)
fig.savefig('confusion_matrix_LUAD_LUSC_XGBClassifier.png')

plt.clf()
fig, ax = plt.subplots(figsize=(5, 5))
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, label="AUC=" + str(auc)[:6], color='red')
auc = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr, tpr, label="AUC=" + str(auc)[:6], color='red')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc='lower right')
fig.savefig('roc_curve_luadlusc_XGBClassifier.png')

