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
from server_pipeline.crop_image import Cropping
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
from Sqldb.SQL import retrieve_embeddings


model = tf.keras.models.load_model('/Users/harshsingh/Desktop/projects/face/model')
FRmodel = model

def img_to_encoding(img, model):
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0) # add a dimension of 1 as first dimension
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)


path_to_db = "/Users/harshsingh/Desktop/projects/face/Recognition_pipeline/Sqldb/face_database.db"
database = retrieve_embeddings(path_to_db)
print(f""""Database retrieved successfully! Number of entries are {len(database)}""")



def who_is_it(img, database, model):
    
    encoding =  img_to_encoding(img, model)

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
cropper = Cropping(test_path, output_path, name , woffset, hoffset)
test_path = cropper.crop_faces()


dis, name = who_is_it(test_path, database, FRmodel)
print(f"Distance: {dis} \n Identity: {name}")