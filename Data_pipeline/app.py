import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2
from cvzone.FaceDetectionModule import FaceDetector


def save_image(image, name):
    """Save the image in the Database folder under the user's name."""
    # cut out the face
    detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)
    image, bboxs = detector.findFaces(image, draw=False)
    if not bboxs:
        st.error("No faces detected in the image.")
        return
    bbox = bboxs[0]
    x, y, w, h = bbox['bbox']
    offset_w = 30
    offset_h = 30
    offsetW = (offset_w / 100) * w
    x = int(x - offsetW)
    w = int(w + offsetW * 2)
    offsetH = (offset_h / 100) * h
    y = int(y - offsetH * 3)
    h = int(h + offsetH * 3.5)
    image = image[y:y + h, x:x + w]

    # save the image
    base_folder = "../Database"
    user_folder = os.path.join(base_folder, name)
    image_name = name+".jpg"
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

    if name:
        col1, col2 = st.columns(2)
        with col1:
            enable = st.checkbox("Enable camera")
            picture = st.camera_input("Take a picture", disabled=not enable)
        with col2:
            if picture:
                save = st.button("Save Image")
                st.image(picture)
                if save:
                    img = Image.open(picture)
                    img_array = np.array(img)
                    saved_image_path = save_image(img_array, name)
                    st.success(f"Image saved to {saved_image_path}")


if __name__ == "__main__":
    main()
