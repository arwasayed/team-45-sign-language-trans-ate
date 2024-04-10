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
from collectData import actions
from train import model
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from collectData import actions ,no_sequences, sequence_length ,DATA_PATH
from tensorflow.keras import layers

model1=model.load_weights('action.weights.h5')
print(model.summary())
def test():
        
    # 1. New detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.3

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with fnc.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = fnc.mediapipe_detection(frame, holistic)
            # print(results)
            
            # # Draw landmarks
            # draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = fnc.extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                
                
            #3. Viz logic
                if np.unique(predictions[-30:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                # image = prob_viz(res, actions, image, colors)
                # image = prob_viz( actions, image, colors)
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    if sys.argv[1] == "test":
        test()
    else:
        print("Invalid argument. Usage: python test.py test")