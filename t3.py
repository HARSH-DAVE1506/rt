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

# PID controller parameters
kp = 0.1  # Proportional gain
ki = 0.01  # Integral gain
kd = 0.05  # Derivative gain

integral_x, integral_y = 0, 0
prev_error_x, prev_error_y = 0, 0

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
            x, y = int(index_finger_tip.x * image.shape[1]), int(index_finger_tip.y * image.shape[0])

            # Draw a circle at the index finger tip
            cv2.circle(image, (x, y), 10, (0, 255, 0), -1)

            # Calculate error
            error_x = x - image.shape[1] / 2
            error_y = y - image.shape[0] / 2

            # Update integral
            integral_x += error_x
            integral_y += error_y

            # Calculate derivative
            derivative_x = error_x - prev_error_x
            derivative_y = error_y - prev_error_y

            # Calculate PID output
            output_x = kp * error_x + ki * integral_x + kd * derivative_x
            output_y = kp * error_y + ki * integral_y + kd * derivative_y

            # Convert output to pan/tilt commands
            pan = -output_x  # Invert for natural panning
            tilt = output_y

            # Clamp values to acceptable range
            pan = max(min(pan, 180), -180)
            tilt = max(min(tilt, 90), -30)

            send_command(pan, tilt)

            # Update previous error
            prev_error_x = error_x
            prev_error_y = error_y

    # Display the image
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Exit on ESC
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
