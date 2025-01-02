import face_recognition
import cv2
import numpy as np
from deepface import DeepFace

def detect_faces(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Unable to load image from '{image_path}'. Please check the file path.")
        exit(1)

    # Convert the image from BGR (OpenCV) to RGB (face_recognition)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find all face locations in the image
    face_locations = face_recognition.face_locations(rgb_image)

    return image, face_locations

def classify_fake_or_real(image_path):
    try:
        # Perform analysis using DeepFace
        result = DeepFace.analyze(image_path, actions=['emotion'], enforce_detection=False, anti_spoofing=True)
        print("Result",result)
        return "Real"
    except Exception as e:
        print(f"Error in DeepFace analysis: {e}")
        return "Fake"

def main(image_path):
    # Detect faces in the image
    image, face_locations = detect_faces(image_path)

    if len(face_locations) == 0:
        print("No faces found in the image.")
    else:
        print(f"Found {len(face_locations)} face(s) in this photograph.")
        for face_location in face_locations:
            top, right, bottom, left = face_location
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        print("Classifying face(s) as real or fake using DeepFace...")
        result = classify_fake_or_real(image_path)
        print("Analysis Result:", result)

        cv2.imshow("Detected Faces", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        

if __name__ == "__main__":
    image_path = "./images/SaranKonala_Photo.jpeg"  # Replace with your image file path
    main(image_path)
