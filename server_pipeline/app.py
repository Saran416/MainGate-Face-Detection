import math
import time
import os
import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cvzone
import warnings
import numpy as np
from Sqldb.SQL import retrieve_embeddings
import tensorflow as tf
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
from cropper_img import Cropping
warnings.filterwarnings("ignore")

def img_to_encoding(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0) # add a dimension of 1 as first dimension
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)

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
        print(f"Closest:  {min_dist} {identity}")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

model = tf.keras.models.load_model('../model')
FRmodel = model

path_to_db = "./Sqldb/face_database.db"
database = retrieve_embeddings(path_to_db)

offset_w=10
offset_h=10
cropper = Cropping(offset_w, offset_h)  

try:
    count = 0
    while True:
        # Read frame from webcam
        success, img = cap.read()
        if not success:
            print("Failed to read from webcam.")
            break
        img = cv2.flip(img, 1)
        count = count + 1
        if(count >10):
            if(cropper.crop_face(img) != False):

                cropped_face, box = cropper.crop_face(img)

                path  = f"./face/img_{count}.jpeg"
                cv2.imwrite(path, cropped_face)
                dis, name = who_is_it(path, database, FRmodel)

                img_with_box = cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 3)
                cv2.imshow("Webcam",img_with_box)
                # time.sleep(1)
            else:
                # time.sleep(1)
                cv2.imshow("Webcam",img)
                pass

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()