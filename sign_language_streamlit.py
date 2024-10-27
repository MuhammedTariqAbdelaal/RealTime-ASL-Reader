#Import libraries
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import streamlit as st

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Preparing real-time hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

# Mapping the predicted output to ASL letters
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
               20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'space'}

# storing the recognized text
recognized_text = ""
last_predicted_time = time.time()

# Streamlit app configuration
st.title('Sign Language Recognition')

st.sidebar.header("Control Panel")
run = st.sidebar.checkbox('Run the Webcam')
stop = st.sidebar.button('Stop Webcam')

# Show instructions if the webcam isn't running
if not run:
    st.header("Welcome to the Sign Language Recognition App!")
    st.markdown("""
        - Click the checkbox to start the webcam.
        - Make hand gestures to see the recognized text.
        - Click 'Stop Webcam' when you're done.
    """)

if run and not stop:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    text_placeholder = st.empty()
    progress_bar = st.progress(0)

    while not stop:
        ret, frame = cap.read()
        if not ret:
            break

        data_aux = []
        x_ = []
        y_ = []

        # Process the frame to detect hand landmarks
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W, _ = frame.shape

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Collect landmark coordinates
            for i in range(21):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize the coordinates
            for i in range(21):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Update the recognized text every 3 seconds
                if time.time() - last_predicted_time > 3:
                    if predicted_character == "del":
                        recognized_text = recognized_text[:-1]
                    elif predicted_character == "space":
                        recognized_text += " "
                    else:
                        recognized_text += predicted_character

                    last_predicted_time = time.time()

                # Draw a bounding box around the hand and display the predicted character
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (225, 225, 225), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (225, 225, 225), 3,
                            cv2.LINE_AA)

        # Update the Streamlit app with the webcam feed and recognized text
        stframe.image(frame, channels="BGR", use_column_width=True, caption="Webcam Feed")
        text_placeholder.markdown(f"<h2 style='color: blue;'>Recognized Text: {recognized_text}</h2>", unsafe_allow_html=True)

        # Update the progress bar
        progress_bar.progress(min(int((time.time() - last_predicted_time) / 3 * 100), 100))

    cap.release()
    cv2.destroyAllWindows()

# Footer
st.markdown("""
    <div style="position: fixed; bottom: 0; width: 100%; text-align: center;">
    <hr>
    <p style="color: grey;">Developed by Ali Ashraf | Sign Language Recognition</p>
    </div>
    """, unsafe_allow_html=True)
