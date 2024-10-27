# Importing the necessary libraries
import pickle         # For loading and saving the pre-trained machine learning model
import cv2            # OpenCV for handling real-time webcam feed and image processing
import mediapipe as mp  # MediaPipe for detecting and extracting hand landmarks
import numpy as np    # NumPy for numerical operations and data manipulation
import time           # To keep track of time for real-time predictions
import streamlit as st  # Streamlit for creating a web-based user interface

# Load the pre-trained ASL recognition model from a pickle file
model_dict = pickle.load(open('./model.p', 'rb'))  # Load the model dictionary from the file
model = model_dict['model']  # Extract the trained model from the dictionary

# Initialize MediaPipe's hand detection and drawing utilities
mp_hands = mp.solutions.hands  # Initialize the hand tracking solution
mp_drawing = mp.solutions.drawing_utils  # Drawing utility for visualizing landmarks
mp_drawing_styles = mp.solutions.drawing_styles  # Predefined drawing styles for consistency

# Configure the hand detection object for real-time use
hands = mp_hands.Hands(
    static_image_mode=False,     # False to use it for real-time video
    max_num_hands=1,             # Detect only one hand at a time
    min_detection_confidence=0.3 # Minimum confidence threshold for hand detection
)

# Define a dictionary to map model predictions to corresponding ASL letters and actions
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'space'
}

# Initialize variables to store recognized text and track the timing of predictions
recognized_text = ""        # To hold the accumulated recognized text
last_predicted_time = time.time()  # Store the last prediction time to control update intervals

# Streamlit web app configuration
st.title('Sign Language Recognition')  # Title of the web app

# Sidebar configuration for controlling webcam behavior
st.sidebar.header("Control Panel")
run = st.sidebar.checkbox('Run the Webcam')  # Checkbox to start/stop the webcam feed
stop = st.sidebar.button('Stop Webcam')      # Button to stop the webcam

# Show usage instructions when the webcam is not running
if not run:
    st.header("Welcome to the Sign Language Recognition App!")
    st.markdown("""
        - Click the checkbox to start the webcam.
        - Make hand gestures to see the recognized text.
        - Click 'Stop Webcam' when you're done.
    """)

# If the webcam is running and has not been stopped
if run and not stop:
    cap = cv2.VideoCapture(0)  # Start video capture from the default webcam (ID 0)
    stframe = st.empty()  # Placeholder for displaying webcam feed in Streamlit
    text_placeholder = st.empty()  # Placeholder for displaying recognized text
    progress_bar = st.progress(0)  # Progress bar for tracking prediction updates

    # Continuously read frames from the webcam until 'stop' is triggered
    while not stop:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            break  # If no frame is captured, exit the loop

        # Variables to store coordinates and features for each frame
        data_aux = []  # Temporary storage for feature data from hand landmarks
        x_ = []        # To store x-coordinates of landmarks for normalization
        y_ = []        # To store y-coordinates of landmarks for normalization

        # Convert the captured frame to RGB as MediaPipe uses RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W, _ = frame.shape  # Get the height and width of the frame

        # Process the frame to detect hand landmarks
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            # If hand landmarks are detected, proceed with the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            # Draw the detected hand landmarks on the frame for visualization
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Collect x and y coordinates of each landmark
            for i in range(21):
                x = hand_landmarks.landmark[i].x  # x-coordinate (normalized 0-1)
                y = hand_landmarks.landmark[i].y  # y-coordinate (normalized 0-1)
                x_.append(x)  # Append x-coordinate for normalization
                y_.append(y)  # Append y-coordinate for normalization

            # Normalize the collected coordinates based on the bounding box
            for i in range(21):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))  # Normalize x-coordinates
                data_aux.append(y - min(y_))  # Normalize y-coordinates

            # If we have the expected number of features (21 landmarks * 2), make a prediction
            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])  # Predict the ASL character
                predicted_character = labels_dict[int(prediction[0])]  # Convert prediction to corresponding ASL letter

                # Update recognized text every 3 seconds
                if time.time() - last_predicted_time > 3:
                    if predicted_character == "del":
                        recognized_text = recognized_text[:-1]  # Remove last character for "del" action
                    elif predicted_character == "space":
                        recognized_text += " "  # Add a space for "space" action
                    else:
                        recognized_text += predicted_character  # Append the recognized character

                    last_predicted_time = time.time()  # Reset the prediction time

                # Draw a bounding box around the detected hand and show the predicted character
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Draw the bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (225, 225, 225), 4)  # White bounding box
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (225, 225, 225), 3,
                            cv2.LINE_AA)  # Display predicted character above the box

        # Update the Streamlit app with the processed webcam frame and recognized text
        stframe.image(frame, channels="BGR", use_column_width=True, caption="Webcam Feed")
        text_placeholder.markdown(f"<h2 style='color: blue;'>Recognized Text: {recognized_text}</h2>", unsafe_allow_html=True)

        # Update the progress bar to indicate when the next character will be recognized
        progress_bar.progress(min(int((time.time() - last_predicted_time) / 3 * 100), 100))

    # Release the webcam and clean up when the loop ends
    cap.release()
    cv2.destroyAllWindows()

# Footer for Streamlit app
st.markdown("""
    <div style="position: fixed; bottom: 0; width: 100%; text-align: center;">
    <hr>
    <p style="color: grey;">Developed by Ali-ElDin Ashraf and Muhammed Aboseif | Sign Language Recognition.................................................</p>
    </div>
    """, unsafe_allow_html=True)
