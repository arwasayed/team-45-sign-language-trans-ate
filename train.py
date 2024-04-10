import sys
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import json
from tqdm import tqdm
import functions as fnc
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from collectData import actions ,no_sequences, sequence_length ,DATA_PATH
from tensorflow.keras import layers


model = Sequential()
model.add(layers.Input(shape=(30,1530)))  # Define input shape here
model.add(layers.LSTM(64, return_sequences=True, activation='relu'))
model.add(layers.LSTM(128, return_sequences=True, activation='relu'))
model.add(layers.LSTM(64, return_sequences=False, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(actions.shape[0], activation='softmax'))


def train():
    label_map = {label:num for num, label in enumerate(actions)}
    sequences, labels = [], [] 
    for action in actions :
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action]) 
        
    x=np.array(sequences)
    y = keras.utils.to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback])

    res=model.predict(X_test)
    model.save_weights('action.weights.h5')
    model.load_weights('action.weights.h5')
    from sklearn.metrics import multilabel_confusion_matrix,accuracy_score
    yhat=model.predict(X_train)
    ytrue =np.argmax(y_train,axis=1).tolist()
    yhat=np.argmax(yhat,axis=1).tolist()
    multilabel_confusion_matrix(ytrue,yhat)
    print(accuracy_score(ytrue,yhat))
if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    else:
        print("Invalid argument. Usage: python train.py train")

