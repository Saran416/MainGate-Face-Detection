import streamlit as st
import cv2
import os
from PIL import Image
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import os
from crop_image import Cropping


def save_image(image, name):
    """Save the image in the Database folder under the user's name."""
    base_folder = "../Database"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    # Create a subfolder for the user
    user_folder = os.path.join(base_folder, name)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    
    # Save the image
    image_name = name+"_withoutcrop.jpg"
    image_path = os.path.join(user_folder, image_name)
    cv2.imwrite(image_path, image)
    return image_path

def main():
    st.title("Face Capture App")
    st.write("This app will capture your face and save it to the database.")

    # Ask for the user's name
    name = st.text_input("Enter your name:")
    if not name:
        st.warning("Please enter your name to proceed.")
        return

    # Start the camera
    if st.checkbox("Start Camera"):
        cap = cv2.VideoCapture(0)
        captured = st.empty()
        pr = st.button("Capture Image")
        st.info("Make sure your face is centered in the frame!")

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access the webcam. Make sure it's connected.")
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            captured.image(frame_rgb, channels="RGB")

            # Detect if the user's face is centered (optional)
            if pr:
                img_path = save_image(frame, name) 
                output_directory = "../Database/"+name
                offset_percentage_w = 30  # 30% width offset
                offset_percentage_h = 30  # 30% height offset
                cropper = Cropping(img_path, output_directory, name, offset_percentage_w, offset_percentage_h)
                cropper.crop_faces()
                st.write("Saved!")
                #saved_image_path = save_image(frame, name)
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
