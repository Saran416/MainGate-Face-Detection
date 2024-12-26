import streamlit as st
import numpy as np
from PIL import Image
import cv2
from crop_image import Cropper
from pymongo import MongoClient
import gridfs
import io


def save_image(image, name):
    """Save the image in the Database folder under the user's name."""
    # cut out the face
    offset_w = 30
    offset_h = 30

    cropper = Cropper(offset_w, offset_h)
    image = cropper.crop(image)
    if image is None:
        st.error("No faces detected in the image.")
        return

    # save the image to MongoDB
    try:
        client = MongoClient('localhost', 27017)
        db = client['face_database']
        collection = db['faces']
        fs = gridfs.GridFS(db)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        byteIO = io.BytesIO()
        image.save(byteIO, format='JPEG')
        byteArr = byteIO.getvalue()
        existing_file = collection.find_one({'name': name})

        if existing_file:
            file_id = existing_file['file_id']
            fs.delete(file_id)
            
            file_id = fs.put(byteArr, filename=name)
            collection.update_one({'name': name}, {'$set': {'file_id': file_id}})
            return "Updated the image in the Database"
        else:
            file_id = fs.put(byteArr, filename=name)
            collection.insert_one({'name': name, 'file_id': file_id})
            return "Saved the image to the Database."

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return

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
                    msg = save_image(img_array, name)
                    if msg:
                        st.success(msg)

if __name__ == "__main__":
    main()
