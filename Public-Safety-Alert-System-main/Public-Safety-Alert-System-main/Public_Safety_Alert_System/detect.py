import cv2
import mediapipe as mp
import numpy as np
import pygame
import streamlit as st
import time

# Initialize MediaPipe Hands and drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Paths to face detection model files
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

# Load the face detection model
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Initialize pygame for alarm sound
pygame.mixer.init()

# Alarm function
def play_alarm():
    pygame.mixer.music.load("mixkit-alert-alarm-1005.wav")  # Replace with the path to your alarm sound
    pygame.mixer.music.play()

# Function to check SOS gesture
def check_sos_gesture(landmarks):
    if len(landmarks) == 2:  # Both hands detected
        palms_facing = 0
        for hand_landmarks in landmarks:
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            if index_finger.y < wrist.y and middle_finger.y < wrist.y:
                palms_facing += 1
        if palms_facing == 2:  # Both hands with palms facing the camera
            return True
    return False

# Streamlit App
st.title("Human Safety System")
st.sidebar.title("Controls")

start_camera = st.sidebar.button("Start Camera")

# Placeholder for video feed
video_placeholder = st.empty()

# Camera function
def run_camera():
    cap = cv2.VideoCapture(0)  # Open webcam
    alarm_triggered = False
    sos_start_time = None
    sos_confirmed = False

    while start_camera:  # Continue running until the user stops
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from webcam.")
            break

        # Flip frame for correct orientation
        frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Draw hand landmarks and check SOS gesture
        landmarks = []
        sos_detected = False
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks.append(hand_landmarks)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if check_sos_gesture(landmarks):
                sos_detected = True

        # Handle SOS detection with 5-second confirmation
        if sos_detected:
            if sos_start_time is None:
                sos_start_time = time.time()  # Start timing SOS gesture
            elif time.time() - sos_start_time >= 5 and not sos_confirmed:
                sos_confirmed = True
                alarm_triggered = True
        else:
            sos_start_time = None  # Reset timer if SOS gesture stops
            sos_confirmed = False

        # Perform face detection
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
        faceNet.setInput(blob)
        detections = faceNet.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Display alert and play alarm if SOS is confirmed
        if sos_confirmed and alarm_triggered:
            cv2.putText(frame, "SOS Detected! Alerting nearby people and emergency services!", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if not pygame.mixer.music.get_busy():  # Play alarm if not already playing
                play_alarm()

        # Display the frame in the app
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame, channels="RGB")
    
    cap.release()

# Handle camera functionality
if start_camera:
    run_camera()
else:
    st.warning("Click 'Start Camera' to begin.")