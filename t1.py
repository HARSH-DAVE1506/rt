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

def map_to_range(value, from_min, from_max, to_min, to_max):
    # Convert the value from one range to another
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled_value = (value - from_min) / from_range
    return to_min + (scaled_value * to_range)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read frame from camera.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    if results.multi_hand_landmarks:
       for hand_landmarks in results.multi_hand_landmarks:
            # Calculate the center of the hand
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
            hand_center_x = sum(x_coords) / len(x_coords)
            hand_center_y = sum(y_coords) / len(y_coords)

            # Map hand position to pan and tilt angles
            pan_degree = map_to_range(hand_center_x, 0, 1, -180, 180)
            tilt_degree = map_to_range(hand_center_y, 0, 1, 90, -30)  # Inverted Y-axis

            # Round to integers
            pan_degree = int(round(pan_degree))
            tilt_degree = int(round(tilt_degree))

            command = {
                "T": 133,
                "X": pan_degree,
                "Y": tilt_degree,
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

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
