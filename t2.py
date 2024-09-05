import cv2
import mediapipe as mp
import numpy as np
import serial  
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import json
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define settings
ACCEPTABLE_X_ERROR = 75
ACCEPTABLE_Y_ERROR = 60

serial_port = '/dev/ttymxc3'  
baud_rate = 115200
ser = serial.Serial(serial_port, baud_rate, timeout=1)

# Initialize the hand tracking model
hands = mp.solutions.hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    static_image_mode=False,
    model_complexity=0,
    max_num_hands=1
)

base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 20)

def flush_buffer():
    ser.reset_output_buffer()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read frame from camera.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    
    frame_height, frame_width = frame.shape[:2]
    frame_center = (int(frame_width/2), int(frame_height/2))
    
    # Draw the acceptable error ellipse
    cv2.ellipse(frame, frame_center, (ACCEPTABLE_X_ERROR, ACCEPTABLE_Y_ERROR), 0, 0, 360, (255,255,255), 1)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate the center of the hand
            x_coords = [landmark.x * frame_width for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y * frame_height for landmark in hand_landmarks.landmark]
            hand_center_x = int(sum(x_coords) / len(x_coords))
            hand_center_y = int(sum(y_coords) / len(y_coords))
            
            # Draw the hand center
            cv2.circle(frame, (hand_center_x, hand_center_y), 5, (0, 0, 255), -1)

            # Calculate offsets of hand center from center of frame
            delta_x = hand_center_x - frame_center[0]
            delta_y = hand_center_y - frame_center[1]

            # Flush the serial buffer
            flush_buffer()

            # Determine motor movements based on error thresholds
            if abs(delta_x) > ACCEPTABLE_X_ERROR:
                pan_direction = 1 if delta_x > 0 else -1
                pan_degree = pan_direction * min(abs(delta_x), 180)
            else:
                pan_degree = 0

            if abs(delta_y) > ACCEPTABLE_Y_ERROR:
                tilt_direction = -1 if delta_y > 0 else 1  # Inverted Y-axis
                tilt_degree = tilt_direction * min(abs(delta_y), 90)
            else:
                tilt_degree = 0

            # Create and send the command
            if pan_degree != 0 or tilt_degree != 0:
                command = {
                    "T": 133,
                    "X": int(pan_degree),
                    "Y": int(tilt_degree),
                    "SPD": 0,
                    "ACC": 0
                }

                json_command = json.dumps(command)
                print("Sending command:", json_command)

                try:
                    ser.write((json_command + '\n').encode('utf-8'))
                    print("Command sent successfully")
                except serial.SerialException as e:
                    print(f'Failed to send command: {e}')

    # Display the frame
    cv2.imshow("Hand Tracking", frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
