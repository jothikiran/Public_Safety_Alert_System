from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import threading

app = Flask(__name__)

# Initialize MediaPipe Hands and drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Paths to face detection model files
faceProto = "models/opencv_face_detector.pbtxt"
faceModel = "models/opencv_face_detector_uint8.pb"

# Load the face detection model
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Initialize pygame for alarm sound
pygame.mixer.init()

# Global variables
alarm_triggered = False
noise_level = 65
gesture_detected = "None"
system_status = {
    "microphone": "Active",
    "camera": "Streaming (720p)",
    "network": "Connected (45ms)",
    "uptime": "00:00:00"
}

# Alarm function
def play_alarm():
    pygame.mixer.music.load("sounds/alarm.wav")
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

def generate_frames():
    global alarm_triggered, gesture_detected
    cap = cv2.VideoCapture(0)
    sos_start_time = None
    sos_confirmed = False

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Flip frame for correct orientation
            frame = cv2.flip(frame, 1)

            # Convert to RGB for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Draw hand landmarks and check SOS gesture
            landmarks = []
            sos_detected = False
            gesture_detected = "None"
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks.append(hand_landmarks)
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                if check_sos_gesture(landmarks):
                    sos_detected = True
                    gesture_detected = "Emergency"

            # Handle SOS detection with 5-second confirmation
            if sos_detected:
                if sos_start_time is None:
                    sos_start_time = time.time()
                elif time.time() - sos_start_time >= 5 and not sos_confirmed:
                    sos_confirmed = True
                    alarm_triggered = True
                    play_alarm()
            else:
                sos_start_time = None
                sos_confirmed = False

            # Perform face detection
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
            faceNet.setInput(blob)
            detections = faceNet.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Display alert if SOS is confirmed
            if sos_confirmed and alarm_triggered:
                cv2.putText(frame, "SOS Detected! Alerting emergency services!", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html', 
                           gesture_detected=gesture_detected,
                           noise_level=noise_level)

@app.route('/security')
def security():
    return render_template('security.html', status=system_status)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gesture_status')
def gesture_status():
    global gesture_detected
    return jsonify({"gesture": gesture_detected})

@app.route('/alarm_status')
def alarm_status():
    global alarm_triggered
    status = alarm_triggered
    alarm_triggered = False  # Reset after reading
    return jsonify({"alarm": status})

if __name__ == '__main__':
    app.run(threaded=True, debug=True)