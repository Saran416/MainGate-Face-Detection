import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace

def detect_face(image):
    try:
        # Use DeepFace to detect faces
        result = DeepFace.extract_faces(image, enforce_detection=False)

        print("Face detection result:", result)
        
        # If result is not empty, extract face information
        if result:
            # Iterate through faces detected
            for face in result:
                # Ensure 'region' exists in face info
                if 'facial_area' in face:
                    x = int(face['facial_area']['x'])
                    y = int(face['facial_area']['y'])
                    w = int(face['facial_area']['w'])
                    h = int(face['facial_area']['h'])
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                else:
                    print("No 'region' key in detected face")
        else:
            print("No faces detected")
            
        return image, result
    except Exception as e:
        print(f"Error in face detection: {e}")
        return image, []


def anti_spoofing(image, faces):
    if not faces:
        print("No faces for anti-spoofing")
        return image

    # Perform anti-spoofing for each detected face
    for face in faces:
        if 'facial_area' in face:
            x = int(face['facial_area']['x'])
            y = int(face['facial_area']['y'])
            w = int(face['facial_area']['w'])
            h = int(face['facial_area']['h'])
            face_img = image[y:y + h, x:x + w]
            try:
                if face_img.size == 0:
                    raise ValueError("Empty face image, skipping analysis.")
                
                # Anti-spoofing analysis (already in your code)
                result = DeepFace.analyze(face_img, enforce_detection=False, anti_spoofing=True)
                cv2.putText(image, "Real", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            except Exception as e:
                print(f"Error in anti-spoofing: {e}")
                cv2.putText(image, "Fake", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            print("No 'region' key in face for anti-spoofing")
    
    return image

 

# Streamlit camera input to capture image
image_file = st.camera_input("Capture an image")

if image_file is not None:
    # Open the image file and convert it to a numpy array
    img = Image.open(image_file)
    img = np.array(img)
    
    # Convert the image from RGB (PIL) to BGR (OpenCV)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Detect face and draw a rectangle around it using DeepFace
    img_with_guideline, faces = detect_face(img_bgr)
    
    # Apply anti-spoofing analysis
    img_with_guideline_and_anti_spoof = anti_spoofing(img_with_guideline, faces)
    
    # Convert back to RGB for Streamlit display
    img_with_guideline_rgb = cv2.cvtColor(img_with_guideline_and_anti_spoof, cv2.COLOR_BGR2RGB)
    
    # Show the image with face guideline and anti-spoofing
    st.image(img_with_guideline_rgb, caption="Image with Face Detection and Anti-Spoofing", use_container_width=True)
