import streamlit as st
import numpy as np
from PIL import Image
import cv2
from crop_image import Cropper
from image_vector import img_to_encoding
from pymongo import MongoClient
import gridfs
import io
import asyncio
import warnings

warnings.filterwarnings("ignore")

client = MongoClient('localhost', 27017)
db_images = client['face_database']
db_vectors = client['face_vectors']
fs = gridfs.GridFS(db_images, "faces")
collection = db_images["faces"]
vector_collection = db_vectors["vectors"]

# Asynchronous save_image function
async def save_image(image, name, update=False):
    """Save or update the image in the database under the user's name."""
    offset_w = 30
    offset_h = 30

    # Crop the face using the Cropper class
    cropper = Cropper(offset_w, offset_h)
    image = cropper.crop(image)
    if image is None:
        st.error("No faces detected in the image.")
        return

    # Save or update the image in Mongodb_images
    try:
        vector = img_to_encoding(image)
        vector = vector[0].tolist()
        st.write("Vector: ", vector)

        # Convert the cropped image to RGB and save it as a binary stream
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        buffer.seek(0)

        # Check if the file already exists
        existing_file = await asyncio.to_thread(collection.find_one, {'name': name})
        if existing_file:
            st.write("An image already exists for this name.")
            # update = st.button("Update Image");
            st.warning("If updated previous data will be lost")
            if update:
                # Update the existing image in GridFS
                existing_file_id = existing_file['file_id']
                await asyncio.to_thread(fs.delete, existing_file_id)  # Delete the old file
                new_file_id = await asyncio.to_thread(fs.put, buffer, filename=name)  # Save the new file
                # Update the metadata to reference the new file_id
                await asyncio.to_thread(collection.update_one, 
                                        {'name': name}, 
                                        {'$set': {'file_id': new_file_id}})
                
                await asyncio.to_thread(vector_collection.update_one, 
                                        {'name': name}, 
                                        {'$set': {'vector': vector}})
                
                return False
            else:
                return True
        else:
            # Save the new image to GridFS
            file_id = await asyncio.to_thread(fs.put, buffer, filename=name)
            # Store metadata in the collection
            await asyncio.to_thread(collection.insert_one, {'name': name, 'file_id': file_id})

            await asyncio.to_thread(vector_collection.insert_one, {'name': name, 'vector': vector})
            # st.success("Saved the image to the database")
            return False

    except Exception as e:
        st.error(f"An error occurred: {e}")
        # print(e)
        return False


# Asynchronous main function
async def main():
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
            picture = st.camera_input("Please make sure the image is well lit", disabled=not enable)

        with col2:
            if picture:
                st.image(picture, caption="Captured Image")
                save = st.button("Save Image")
                if save:
                    img = Image.open(picture)
                    img_array = np.array(img)

                    # Call the asynchronous save_image function
                    exists = await save_image(img_array, name)

                    if exists:
                        update = st.button("Update Image")
                        if update:
                            await save_image(img_array, name, update=True)
                            st.success("Updated the image in the database")
                    else:
                        st.success("Saved the image to the database")


if __name__ == "__main__":
    # Streamlit doesn't natively support async, so we use asyncio.run to execute the async main function
    asyncio.run(main())
