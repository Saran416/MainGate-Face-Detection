# import the necessary packages
from imutils import face_utils
import dlib
import cv2
import time
import numpy as np

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    # Compute the vertical distances
    A = np.linalg.norm(eye[1] - eye[5])  # Distance between landmark 2 and 6
    B = np.linalg.norm(eye[2] - eye[4])  # Distance between landmark 3 and 5
    # Compute the horizontal distance
    C = np.linalg.norm(eye[0] - eye[3])  # Distance between landmark 1 and 4
    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate Mouth Aspect Ratio (MAR)
def calculate_mar(mouth):
    # Compute the vertical distances
    A = np.linalg.norm(mouth[2] - mouth[10])  # Distance between upper lip and lower lip (landmark 63 and 67)
    B = np.linalg.norm(mouth[4] - mouth[8])   # Distance between upper lip and lower lip (landmark 61 and 65)
    # Compute the horizontal distance
    C = np.linalg.norm(mouth[0] - mouth[6])  # Distance between corners of the mouth (landmark 60 and 64)
    # MAR formula
    mar = (A + B) / (2.0 * C)
    return mar

# Path to the facial landmark predictor
p = "./spoof_detection/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)

prev = 0
next = 0
fps = 0
num = 0

while True:
    # Capture frame from the webcam
    success, image = cap.read()
    if not success:
        print("Failed to read from webcam.")
        break

    cv2.imwrite("test.jpg", image)
    image = cv2.flip(image, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    rects = detector(gray, 0)

    for (i, rect) in enumerate(rects):
        # Get facial landmarks
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # Extract the coordinates for the left and right eyes
        left_eye = shape[42:48]  # Indices for the left eye
        right_eye = shape[36:42]  # Indices for the right eye

        # Calculate EAR for both eyes
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        # Average the EAR for both eyes
        ear = (left_ear + right_ear) / 2.0

        # Draw the eye landmarks
        # for (x, y) in np.concatenate((left_eye, right_eye)):
            # cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # Extract the coordinates for the mouth
        mouth = shape[48:68]  # Indices for the mouth (outer lips)

        # Calculate MAR for the mouth
        mar = calculate_mar(mouth)

        # Draw the mouth landmarks
        # for (x, y) in mouth:
            # cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

        # Display EAR and MAR on the frame
        cv2.putText(image, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(image, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Show the output image
    cv2.imshow("Output", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Average FPS:", fps / num)
        break

    # Calculate FPS
    prev = next
    next = time.time()
    fps += 1 / (next - prev)
    num += 1

# Release resources
cv2.destroyAllWindows()
cap.release()
