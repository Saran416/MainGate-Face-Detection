from imutils import face_utils
import dlib
import numpy as np
import cv2
import copy

class Checker():
    def __init__(self, predictor_path="./spoof_detection/shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.blink_counter = 0
        self.prev_ear = 0

    # Function to calculate Eye Aspect Ratio (EAR)
    def calculate_ear(self, eye):
        # Compute the vertical distances
        A = np.linalg.norm(eye[1] - eye[5])  # Distance between landmark 2 and 6
        B = np.linalg.norm(eye[2] - eye[4])  # Distance between landmark 3 and 5
        # Compute the horizontal distance
        C = np.linalg.norm(eye[0] - eye[3])  # Distance between landmark 1 and 4
        # EAR formula
        ear = (A + B) / (2.0 * C)
        return ear
    
    # Function to calculate Mouth Aspect Ratio (MAR)
    def calculate_mar(self, mouth):
        # Compute the vertical distances
        A = np.linalg.norm(mouth[2] - mouth[10])  # Distance between upper lip and lower lip (landmark 63 and 67)
        B = np.linalg.norm(mouth[4] - mouth[8])   # Distance between upper lip and lower lip (landmark 61 and 65)
        # Compute the horizontal distance
        C = np.linalg.norm(mouth[0] - mouth[6])  # Distance between corners of the mouth (landmark 60 and 64)
        # MAR formula
        mar = (A + B) / (2.0 * C)
        return mar
    
    def idle(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = self.detector(gray, 0)
        if not faces:
            # print("No faces found in the image.")
            return image, False, []

        
        # Process only the first detected face
        face = faces[0]

        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())

        # Crop the face from the original image
        cropped_face = copy.deepcopy(image[y:y+h, x:x+w])

        # Get the facial landmarks
        shape = self.predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (255, 255, 255), -1)

        # Extract the coordinates for the left and right eyes
        left_eye = shape[42:48]  # Indices for the left eye
        right_eye = shape[36:42]  # Indices for the right eye

        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)

        # Average the EAR for both eyes
        ear = (left_ear + right_ear) / 2.0

        # Check for blink
        if(self.prev_ear - ear > 0.08):
            self.blink_counter += 1
            print("Blink detected")

        self.prev_ear = ear

        # Extract the coordinates for the mouth
        # mouth = shape[48:68]  # Indices for the mouth (outer lips)

        # # Calculate MAR for the mouth
        # mar = self.calculate_mar(mouth)

        # Display EAR and MAR on the frame
        # cv2.putText(image, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # cv2.putText(image, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        # cv2.imwrite("face.jpg", cropped_face)
        return image,True, cropped_face

    def check_spoof(self):
        if self.blink_counter > 0:
            print("Real face.")
            self.blink_counter = 0
            return True

        return False
