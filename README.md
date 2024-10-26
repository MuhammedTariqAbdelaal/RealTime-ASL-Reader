# Sign Language Recognition

This project is a private Sign Language Recognition application that uses Python and machine learning to recognize American Sign Language (ASL) letters from a live webcam feed.

## Project Overview

The application detects hand gestures and converts them into text using a trained machine learning model. It relies on MediaPipe for hand landmark detection and Streamlit for a simple web-based interface.

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
