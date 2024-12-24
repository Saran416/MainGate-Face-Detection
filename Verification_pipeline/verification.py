import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from crop_image import Cropping
K.set_image_data_format('channels_last')
import numpy as np
from numpy import genfromtxt
import PIL
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import os
from tensorflow.keras.models import model_from_json
import time




model = tf.keras.models.load_model('/Users/harshsingh/Desktop/projects/face/model')
FRmodel = model

def img_to_encoding(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0) # add a dimension of 1 as first dimension
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)


database = {}
base_dir = "/Users/harshsingh/Desktop/projects/face/indian celebrities dataset/cropped_data"


time_build_databse = time.time()
for name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, name)
    
    # Ensure the path is a directory
    if os.path.isdir(folder_path):
        database[name] = []  # Initialize an empty list for the name
        count = 0  # Reset count for each person
        
        for image in os.listdir(folder_path):
            if count == 3:  # Stop after 5 images
                break
            
            img_path = os.path.join(folder_path, image)
            
            # Ensure the path is a valid file
            if os.path.isfile(img_path):
                # Process the image to get its embedding
                embedding = img_to_encoding(img_path, FRmodel)
                database[name].append(embedding)  # Add embedding to the list
                count += 1  # Increment the count

time_build_databse = time.time() - time_build_databse
print(f"Time taken to build the database: {time_build_databse} seconds")
print(f""""Database created successfully! Number of entries are {len(database)}""")

def verify(image_path, identity, database, model):
    
    encoding = img_to_encoding(image_path, model)
    dist = 100
    for embeddings in database[identity]:
        dist = min(np.linalg.norm(tf.subtract(embeddings, encoding)), dist)
    if dist < 0.9:
        print("It's " + str(identity) + ", welcome in!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False
    print(dist)
    return dist, door_open

img_path = "/Users/harshsingh/Desktop/projects/face/indian celebrities dataset/cropped_data/alok nath/62749_kweylwdatv_1499690485.jpg_cropped.jpg"
identity = "alok nath"

woffset = 0
hoffset = 0
output_path = "/Users/harshsingh/Desktop/projects/face/"
cropper = Cropping(img_path, output_path, identity , woffset, hoffset)
test_path = cropper.crop_faces()

verify(img_path, identity, database, FRmodel)