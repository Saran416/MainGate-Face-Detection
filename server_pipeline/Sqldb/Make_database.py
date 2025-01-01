from SQL import create_table, insert_embedding
import os
import tensorflow as tf
import numpy as np
import time 


create_table()
base_dir = "/Users/harshsingh/Desktop/projects/face/indian celebrities dataset/cropped_data"

model = tf.keras.models.load_model('/Users/harshsingh/Desktop/projects/face/model')
FRmodel = model

def img_to_encoding(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0) # add a dimension of 1 as first dimension
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)

time1 = time.time()
for name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, name)
    if os.path.isdir(folder_path):
        count = 0
        for image in os.listdir(folder_path):
            if count >= 3:  # Store up to 3 embeddings per person
                break
            img_path = os.path.join(folder_path, image)
            if os.path.isfile(img_path):
                embedding = img_to_encoding(img_path, FRmodel)
                insert_embedding(name, embedding)
                count += 1

print("Time taken to insert embeddings: ", time.time() - time1)
print("Database created successfully!")

