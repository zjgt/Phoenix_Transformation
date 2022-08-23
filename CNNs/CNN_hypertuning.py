import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
from tensorflow.keras import Model
import matplotlib.image as mpimg
import zipfile
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import numpy as np
import keras_tuner
import pathlib
import glob
from sklearn.model_selection import train_test_split



base_dir = 'Ascending2_Low_Removed/'
luad = glob.glob(base_dir + 'LUAD/*.*')
lusc = glob.glob(base_dir + 'LUSC/*.*')

data = []
labels = []

for i in luad:
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb',
    target_size= (512,512))
    image=np.array(image)
    data.append(image)
    labels.append(0)
for i in lusc:
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb',
    target_size= (512,512))
    image=np.array(image)
    data.append(image)
    labels.append(1)

data = np.array(data)
labels = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
# Print the shapes of the data.
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

def build_model(hp):
    inputs = keras.Input(shape=(512, 512, 3))
    # Model type can be MLP or CNN.
    model_type = hp.Choice("model_type", ["mlp", "cnn"])
    x = inputs
    if model_type == "mlp":
        x = layers.Flatten()(x)
        # Number of layers of the MLP is a hyperparameter.
        for i in range(hp.Int("mlp_layers", 1, 3)):
            # Number of units of each layer are
            # different hyperparameters with different names.
            output_node = layers.Dense(
                units=hp.Int(f"units_{i}", 32, 128, step=32), activation="relu",
            )(x)
    else:
        # Number of layers of the CNN is also a hyperparameter.
        for i in range(hp.Int("cnn_layers", 1, 3)):
            x = layers.Conv2D(
                hp.Int(f"filters_{i}", 32, 128, step=32),
                kernel_size=(3, 3),
                activation="relu",
            )(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Flatten()(x)

    # A hyperparamter for whether to use dropout layer.
    if hp.Boolean("dropout"):
        x = layers.Dropout(0.2)(x)

    # The last layer contains 10 units,
    # which is the same as the number of classes.
    outputs = layers.Dense(units=1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model.
    model.compile(
        loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam",
    )
    return model

# Initialize the `HyperParameters` and set the values.
hp = keras_tuner.HyperParameters()
hp.values["model_type"] = "cnn"
# Build the model using the `HyperParameters`.
model = build_model(hp)
# Test if the model runs with our data.
model(x_train[:100])
# Print a summary of the model.
model.summary()

tuner = keras_tuner.RandomSearch(
    build_model,
    max_trials=20,
    # Do not resume the previous search in the same directory.
    overwrite=True,
    objective="val_accuracy",
    # Set a directory to store the intermediate results.
    directory="ML_Practice/LUAD_LUSC_Training_Results/Hyper/",
)

tuner.search(
    x_train,
    y_train,
    validation_split=0.3,
    epochs=15,
    # Use the TensorBoard callback.
    # The logs will be write to "/tmp/tb_logs".
    callbacks=[keras.callbacks.TensorBoard("ML_Practice/LUAD_LUSC_Training_Results/Hyper/tb_logs")],
)




