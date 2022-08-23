import os
import pickle
import numpy as np
import seaborn as sns
from plot_keras_history import show_history, plot_history
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

hist_dir = 'Training_History/'
histlist = [f for f in os.listdir(hist_dir) if f.endswith('.npy')]
histories = []
iter = len(histlist)
palette = list(reversed(sns.color_palette("Spectral_r", iter).as_hex()))

for file in histlist:
    file = hist_dir + file
    history = np.load(file, allow_pickle='TRUE').item()
    histories = np.append(histories, history)

plt.clf()
fig, ax = plt.subplots(figsize=(5, 5))


for i in range(iter):
    plt.plot(histories[i]['accuracy'], color=palette[i])
    plt.plot(histories[i]['val_accuracy'], color=palette[i], linestyle=':')
    #The labelcolor argument does not change the line color, need to figure out.
    legends = plt.legend([f'Train_{histlist[i]}', f'Test_{histlist[i]}'], loc=(0.5, 0.6-0.1*i), labelcolor=palette[i])
    plt.gca().add_artist(legends)
    print(histlist[i])
plt.ylim(0.5, 1.0)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='lower right')
plt.savefig(os.path.join(hist_dir, 'History_accuracy_comparison.png'))
plt.show()

# Plot training and validation loss
plt.clf()
fig, ax = plt.subplots(figsize=(5, 5))
for i in range(iter):
    plt.plot(histories[i]['loss'], color=palette[i])
    plt.plot(histories[i]['val_loss'], color=palette[i], linestyle=':')
    legends = plt.legend([f'Train_{histlist[i]}', f'Test_{histlist[i]}'], loc=(0.5, 0.8-0.1*i), labelcolor=palette[i])
    plt.gca().add_artist(legends)
    print(histlist[i])

plt.ylim(0, 2.0)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='lower right')
plt.savefig(os.path.join(hist_dir, 'History_loss_comparison.png'))
plt.show()




