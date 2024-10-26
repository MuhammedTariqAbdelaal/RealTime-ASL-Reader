#Import libraries
import os
import pickle
import mediapipe as mp
import cv2

# Preparing hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Path to data
DATA_DIR = './data'

data = []
labels = []

# Loop through each subfolder in the data directory
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        # Read the image
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract the landmarks
                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize the landmarks based on the bounding box
                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Only add the data if it has the expected number of features
            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(dir_)

# Save the processed data
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
