# Import necessary libraries
import os           # For interacting with the operating system and handling file paths
import pickle       # For saving and loading data in a serialized format
import mediapipe as mp  # MediaPipe for hand detection and extracting landmarks
import cv2          # OpenCV for reading and processing images

# Prepare MediaPipe's hand detection module
mp_hands = mp.solutions.hands  # Load the hands module from MediaPipe
hands = mp_hands.Hands(
    static_image_mode=True,           # Process images statically (not in a real-time feed)
    min_detection_confidence=0.3      # Minimum confidence threshold for detecting hands
)

# Define the path to the directory containing the image dataset
DATA_DIR = './data'

# Initialize empty lists to store feature data and their corresponding labels
data = []    # List to store the extracted features from each image
labels = []  # List to store the corresponding labels (gesture type) for each image

# Loop through each subdirectory in the main data directory
# Each subdirectory should represent a class label (e.g., A, B, C for ASL letters)
for dir_ in os.listdir(DATA_DIR):
    # Loop through each image file in the current subdirectory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Temporary list to store features for the current image
        x_ = []        # List to store x-coordinates of landmarks for normalization
        y_ = []        # List to store y-coordinates of landmarks for normalization

        # Read the image using OpenCV
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))  # Load the image file
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB (MediaPipe uses RGB format)

        # Process the image to detect hand landmarks
        results = hands.process(img_rgb)  # Analyze the image for any hand landmarks
        if results.multi_hand_landmarks:
            # If landmarks are detected, process each detected hand (in this case, assuming one hand)
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract the x and y coordinates of all 21 landmarks
                for i in range(21):  # MediaPipe detects 21 landmarks for each hand
                    x = hand_landmarks.landmark[i].x  # Normalized x-coordinate (range 0 to 1)
                    y = hand_landmarks.landmark[i].y  # Normalized y-coordinate (range 0 to 1)
                    x_.append(x)  # Collect x-coordinates for normalization
                    y_.append(y)  # Collect y-coordinates for normalization

                # Normalize the coordinates based on the bounding box created from the detected landmarks
                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    # Normalize x and y coordinates by subtracting the minimum value from the set of detected points
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Only add the data if it has the expected number of features (21 landmarks * 2 = 42 features)
            if len(data_aux) == 42:
                data.append(data_aux)  # Append the normalized features to the data list
                labels.append(dir_)    # Append the corresponding label for the image

# Save the processed feature data and labels into a pickle file for future use
with open('data.pickle', 'wb') as f:
    # Serialize the data and labels into the 'data.pickle' file
    pickle.dump({'data': data, 'labels': labels}, f)
