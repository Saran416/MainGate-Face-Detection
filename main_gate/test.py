import cv2
import dlib
import numpy as np
from scipy.spatial import distance

# Load the pre-trained face detector and facial landmarks predictor
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define the eye aspect ratio (EAR) to detect blinks
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load the video capture
cap = cv2.VideoCapture(0)

# Set up blink threshold and counter
blink_threshold = 0.3
frame_blink_counter = 0
blink_counter = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_detector(gray)

    # Loop over each face detected
    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)

        # Get the coordinates of the eyes
        left_eye = []
        right_eye = []
        for i in range(36, 42):  # Left eye landmarks
            left_eye.append((landmarks.part(i).x, landmarks.part(i).y))
        for i in range(42, 48):  # Right eye landmarks
            right_eye.append((landmarks.part(i).x, landmarks.part(i).y))

        left_eye = np.array(left_eye)
        right_eye = np.array(right_eye)

        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Compute the average EAR
        ear = (left_ear + right_ear) / 2.0

        # Check if blink is detected based on EAR
        if ear < blink_threshold:
            frame_blink_counter += 1
        else:
            if frame_blink_counter >= 3:  # Blink detected
                blink_counter += 1
            frame_blink_counter = 0

        # If no blink is detected, flag as potential spoofing
        if blink_counter == 0:
            cv2.putText(frame, "Spoofing Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw the bounding box around the face
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        
        # Draw landmarks (optional)
        for n in range(36, 48):
            x, y = landmarks.part(n).x, landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # Display the resulting frame
    cv2.imshow("Face Detection with Anti-Spoofing", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
