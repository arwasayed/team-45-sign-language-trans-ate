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
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical


DATA_PATH = os.path.join('MP_Data') 

    # Actions that we try to detect
actions = np.array(['bad','fine','nice','yes','learn','bath','dad','food','stop','sad','I am','no thing','name','what is','your','to meet you','no','bed','mam','help','drink','sorry','thanks'])
newActions = np.array([])
    # Thirty videos worth of data
no_sequences = 30
    # Videos are going to be 30 frames in length
sequence_length = 30
actions=np.append(actions,newActions)
def collectData():
    for action in newActions: 
        for sequence in range(no_sequences):
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with fnc.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # NEW LOOP
        # Loop through actions
        for action in actions:
            # Loop through sequences aka videos
            for sequence in range(no_sequences):
                # Loop through video length aka sequence length
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
        
                        # Make detections
                    image, results =fnc.mediapipe_detection(frame, holistic)
                        #   print(results)
        
                        # Draw landmarks
                    fnc.draw_styled_landmarks(image, results)
                        
                        # NEW Apply wait logic
                    if frame_num == 0: 
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(50)
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        
                        # NEW Export keypoints
                    keypoints = fnc.extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collectData()
    else:
        print("Invalid argument. Usage: python collectData.py collect")

