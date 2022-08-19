import pandas as pd
import numpy as np
from numpy import loadtxt
from numpy import asarray
from PIL import Image
import os

src_dir = 'GAN_4fold/G57_1/'
df = pd.read_csv('Decon_template1500.csv', sep=',', header=0, index_col=0)
length, width = df.shape
n_samples = round((width - 6) / 4)
print(df.shape)
print(df.iloc[0, 10])
print(df.iloc[0, 0])
filenames = [file for file in os.listdir(src_dir) if file.endswith('.png')]
n_samples = len(filenames)
print(n_samples)
n_samples = 100
QRef = open('RGBQuantificaiton_1.csv', 'rb')
QRef = loadtxt(QRef, delimiter=",")
QRef = np.asarray(QRef)
arr = []
for index in range(100):
    name = filenames[index]
    arr.append(name)
    image0 = Image.open(src_dir + name)
    image0 = np.asarray(image0)
    data_R = image0[:, :, 0]
    data_G = image0[:, :, 1]
    data_B = image0[:, :, 2]
    print(name)
    print(index)
    for idx in range(0, length):
        m = round(df.iloc[idx, 2])
        n = round(df.iloc[idx, 3])
        p = round(df.iloc[idx, 4])
        q = round(df.iloc[idx, 5])
        x1 = 6 + index
        x2 = 6 + n_samples + index
        x3 = 6 + 2 * n_samples + index
        x4 = 6 + 3 * n_samples + index
        df.iloc[idx, x1] = round(np.mean(data_R[q:p, m:n]))
        df.iloc[idx, x2] = round(np.mean(data_G[q:p, m:n]))
        df.iloc[idx, x3] = round(np.mean(data_B[q:p, m:n]))

    for i in range(0, length):
        if df.iloc[i, x3] < 10:
            mae = abs(df.iloc[i, x1] - QRef[0, 0]) + abs(df.iloc[i, x2] - QRef[0, 1])
            k = 0
            for j in range(1, 64):
                mae1 = abs(df.iloc[i, x1] - QRef[j, 0]) + abs(df.iloc[i, x2] - QRef[j, 1])
                if mae1 < mae:
                    k = j
                    mae = mae1

            df.iloc[i, x4] = QRef[k, 3]

        elif df.iloc[i, x3] <= 83:
            mae = abs(df.iloc[i, x3] - QRef[63, 2]) + abs(df.iloc[i, x1] - QRef[63, 0])
            k = 63
            for j in range(64, 128):
                mae1 = abs(df.iloc[i, x3] - QRef[j, 2]) + abs(df.iloc[i, x1] - QRef[j, 0])
                if mae1 < mae:
                    k = j
                    mae = mae1

            df.iloc[i, x4] = QRef[k, 3]

        elif df.iloc[i, x3] < 253:
            mae = abs(df.iloc[i, x3] - QRef[127, 2]) + abs(df.iloc[i, x2] - QRef[127, 1])
            k = 127
            for j in range(128, 255):
                mae1 = abs(df.iloc[i, x3] - QRef[j, 2]) + abs(df.iloc[i, x2] - QRef[j, 1])
                if mae1 < mae:
                    k = j
                    mae = mae1

            df.iloc[i, x4] = QRef[k, 3]

        else:
            mae = abs(df.iloc[i, x1] - QRef[185, 0]) + abs(df.iloc[i, x2] - QRef[185, 1])
            k = 185
            for j in range(186, 255):
                mae1 = abs(df.iloc[i, x1] - QRef[j, 0]) + abs(df.iloc[i, x2] - QRef[j, 1])
                if mae1 < mae:
                    k = j
                    mae = mae1

            df.iloc[i, x4] = QRef[k, 3]



df.to_csv(src_dir + 'G57_1_GAN_reconstructed_max.csv', sep=',', header=0)
with open(src_dir + "G57_1_GAN_reconstructed_list_max.txt", "w") as output:
    output.write(str(arr))