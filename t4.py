import cv2
import mediapipe as mp
import numpy as np
import serial
import json
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize serial connection
serial_port = '/dev/ttymxc3'
baud_rate = 115200
ser = serial.Serial(serial_port, baud_rate, timeout=1)

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Pan-Tilt ranges
PAN_MIN, PAN_MAX = -180, 180
TILT_MIN, TILT_MAX = -30, 90

def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def send_command(pan, tilt):
    command = {
        "T": 133,
        "X": int(pan),
        "Y": int(tilt),
        "SPD": 0,
        "ACC": 0
    }
    json_command = json.dumps(command)
    try:
        ser.write((json_command + '\n').encode('utf-8'))
        print(f"Sent: Pan {pan}, Tilt {tilt}")
    except serial.SerialException as e:
        print(f'Failed to send command: {e}')

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to read frame")
        break

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find hand landmarks
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the position of the index finger tip (landmark 8)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = index_finger_tip.x, index_finger_tip.y

            print(f"Hand X: {x:.2f}, Hand Y: {y:.2f}")

            # Map x and y to pan and tilt using a more precise mapping function
            pan = map_range(x, 0, 1, PAN_MIN, PAN_MAX)
            tilt = map_range(y, 0, 1, TILT_MIN, TILT_MAX)

            print(f"Pan: {pan:.2f}, Tilt: {tilt:.2f}")

            send_command(pan, tilt)

            # Draw a circle at the index finger tip
            cv2.circle(image, (int(x * image.shape[1]), int(y * image.shape[0])), 10, (0, 255, 0), -1)

            # Draw debug info on the image
            cv2.putText(image, f"Hand X: {x:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Hand Y: {y:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Pan: {pan:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Tilt: {tilt:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    else:
        # If no hand is detected, display a message
        cv2.putText(image, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the image
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Exit on ESC
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
