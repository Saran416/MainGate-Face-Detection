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



def who_is_it(image_path, database, model):
    
    encoding =  img_to_encoding(image_path, model)

    min_dist = 1000
    identity = None
    for (name, db_enc) in database.items():
        for embeddings in db_enc:
            dist = np.linalg.norm(tf.subtract(embeddings, encoding))

            if dist < min_dist:
                min_dist = dist
                identity = name

    if min_dist > 0.95:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity


test_path = "/Users/harshsingh/Desktop/projects/face/test.jpeg"
output_path = "/Users/harshsingh/Desktop/projects/face"
woffset = 0
hoffset = 0
name = "Ranbir_Kapoor_1"

time_croping = time.time()
cropper = Cropping(test_path, output_path, name , woffset, hoffset)
test_path = cropper.crop_faces()

time_croping = time.time() - time_croping 
print(f"Time taken to crop the image: {time_croping} seconds")

time_find_identity = time.time()
dis, name = who_is_it(test_path, database, FRmodel)

time_find_identity = time.time() - time_find_identity
print(f"Time taken to find the identity: {time_find_identity} seconds")
print(f"Distance: {dis} \n Identity: {name}")


test_path = "/Users/harshsingh/Desktop/projects/face/test.jpeg"
output_path = "/Users/harshsingh/Desktop/projects/face"
woffset = 10
hoffset = 10
name = "Ranbir_Kapoor_2"

time_croping = time.time()
cropper = Cropping(test_path, output_path, name , woffset, hoffset)
test_path = cropper.crop_faces()

time_croping = time.time() - time_croping 
print(f"Time taken to crop the image: {time_croping} seconds")

time_find_identity = time.time()
dis, name = who_is_it(test_path, database, FRmodel)

time_find_identity = time.time() - time_find_identity
print(f"Time taken to find the identity: {time_find_identity} seconds")
print(f"Distance: {dis} \n Identity: {name}")


test_path = "/Users/harshsingh/Desktop/projects/face/test.jpeg"
output_path = "/Users/harshsingh/Desktop/projects/face"
woffset = 15
hoffset = 15
name = "Ranbir_Kapoor_3"

time_croping = time.time()
cropper = Cropping(test_path, output_path, name , woffset, hoffset)
test_path = cropper.crop_faces()

time_croping = time.time() - time_croping 
print(f"Time taken to crop the image: {time_croping} seconds")

time_find_identity = time.time()
dis, name = who_is_it(test_path, database, FRmodel)

time_find_identity = time.time() - time_find_identity
print(f"Time taken to find the identity: {time_find_identity} seconds")
print(f"Distance: {dis} \n Identity: {name}")
