# Sign Language Recognition

A computer vision project focused on bridging the communication gap between hearing and deaf individuals. The project was developed using computer vision techniques, with the goal of creating a robust system capable of recognizing and interpreting American Sign Language (ASL) in real-time from a live webcam feed.


## Project Overview

The application detects hand gestures and converts them into text using a trained RandomForest machine learning model. It relies on MediaPipe for hand landmark detection to get (x, y) coordinates for the hand gesture and Streamlit for a simple web-based interface.

## How it works

1.  Data Preparation(```create_dataset.py```):
    - Extracts hand landmarks from images in the dataset.
    - Saves the processed data to data.pickle for training.

1.  Model Training (```train_classifier.py```):
    - Loads the preprocessed data and trains a Random Forest model.
    - Performs hyperparameter tuning for better accuracy.
    - Saves the best model as ```model.h5```.

1.  Real-Time Prediction (```sign_language_streamlit.py```):
    - Uses a webcam feed to detect hand gestures.
    - Predicts the corresponding ASL letter and displays the recognized text.
  
# Running the app
       streamlit run sign_language_streamlit.py
   - Start the webcam from the Streamlit app.
   - Perform hand gestures in front of the camera to see the detected letters.

# Data used
   - Data used in this project are found in "data" folder.
   - Each categorey has its own folder in "data" folder.
   - The categories correspondants are as follows:
       - 0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
         10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
         19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'space'
