import os
import cv2
from sklearn.cluster import KMeans
import numpy as np
import tensorflow as tf
from pymongo import MongoClient
import pickle
from time import time

import warnings
warnings.filterwarnings("ignore")

class Fetcher():
    def __init__(self, port=27017, db_name='faces', kmeans_clusters=5, data_path='../data_generation/cropped_data', model_path='../model', min_similarity=0.85, load_vectors=True, pkl_path='./img_vectors.pkl'):
        self.client = MongoClient('localhost', port)
        self.db = self.client[db_name]
        self.clusters_size = kmeans_clusters
        self.model = tf.keras.models.load_model(model_path)
        if load_vectors:
            self.img_vectors = self.load_vectors(data_path)
        else:
            self.img_vectors = self.load_vectors_pkl(pkl_path)
            print("Vectors loaded from", pkl_path)
        print(f"Loaded {len(self.img_vectors)} images")
        self.kmeans = self.train_model(self.img_vectors, clusters=kmeans_clusters)
        print("KMeans trained")
        self.min_similarity = min_similarity

    # save the vectors to a pickle file
    def save_vectors(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.img_vectors, f)
        print("Vectors saved to", path)

    # load the vectors from a pickle file
    def load_vectors_pkl(self, path):
        with open(path, 'rb') as f:
            img_vectors = pickle.load(f)
        return img_vectors
    
    # close the connection
    def __del__(self):
            self.client.close()
            print("MongoDB connection closed")

    # save the images to the database
    def save_to_db(self):
        db = self.db
        img_vectors = self.img_vectors
        for img in img_vectors:
            class_name = str(self.kmeans.predict([img['encoding']])[0])
            collection = db[class_name]
            collection.insert_one({'name': img['name'], 'encoding': img['encoding'].tolist()})
        print("Database updated")

    # return the encoding of the image
    def img_to_encoding(self, img_array):
        try:
            img_array = cv2.resize(img_array, (160, 160))
            img_array = np.around(img_array / 255.0, decimals=12)
            img_array = np.expand_dims(img_array, axis=0)
            embedding = self.model.predict_on_batch(img_array)
            return embedding / np.linalg.norm(embedding, ord=2)
        except Exception as e:
            print("Error in img_to_encoding:", e)
            return None

    # load the images from the data_path and return the encodings (temporarily dataset dir for testing)
    def load_vectors(self, data_path):
        img_vectors = []
        for dir in os.listdir(data_path):
            dir_path = os.path.join(data_path, dir)
            if not os.path.isdir(dir_path):
                continue
            for file in os.listdir(dir_path):
                img_path = os.path.join(dir_path, file)
                img_array = cv2.imread(img_path)
                if img_array is None:
                    print(f"Error reading {img_path}, skipping...")
                    continue
                encoding = self.img_to_encoding(img_array)
                img_vectors.append({'name': dir, 'encoding': encoding.flatten()})
        return img_vectors

    # train the kmeans model
    def train_model(self, img_vectors, clusters = 5):
        X = [img['encoding'] for img in img_vectors]
        kmeans = KMeans(n_clusters=clusters, random_state=0)
        kmeans.fit(X)
        return kmeans

    # check if the image is in the collection
    def check_in_collection(self, img):
        db = self.db
        img_vector = self.img_to_encoding(img)
        if img_vector is None:
            return None
        img_vector = [img_vector.flatten()]
        class_name = str(self.kmeans.predict(img_vector)[0])
        collection = db[class_name]
        vectors = collection.find({})
        img_vector = np.array(img_vector)
        for vector in vectors:
            if np.linalg.norm(np.array(vector['encoding']) - img_vector) < self.min_similarity:
                return vector['name'], class_name
        return None, class_name

    # check if the image is in the database
    def check_in_database(self, img, class_found):
        db = self.db
        img_vector = self.img_to_encoding(img)
        if img_vector is None:
            return None
        img_vector = img_vector.flatten()
        img_vector = np.array(img_vector)
        for i in range(self.clusters_size):
            class_name = str(i)
            if class_name == class_found:
                continue
            collection = db[class_name]
            vectors = collection.find({})
            for vector in vectors:
                if np.linalg.norm(np.array(vector['encoding']) - img_vector) < self.min_similarity:
                    return vector['name']
        return None

    # fetch the name of the image (optimized)
    def fetch_name(self, img):
        start = time()
        name, class_name = self.check_in_collection(img)
        if not name:
            print("Miss in collection")
            name = self.check_in_database(img, class_name)
            if not name:
                print("Miss in database")
            else:
                print("Hit in database")
        else:
            print("Hit in collection")
        print("Time taken:", time() - start)
        print('')
        return name

    # save the image to the unoptimized database
    def save_to_unoptimized_db(self):
        db = self.client['unoptimized_image_db']
        img_vectors = self.img_vectors
        collection = db['vectors']
        for img in img_vectors:
            collection.insert_one({'name': img['name'], 'encoding': img['encoding'].tolist()})
        print("Database updated")

    # fetch the name of the name (without optimization)
    def fetch_name_unoptimized(self, img):
        start = time()
        db = self.client['unoptimized_image_db']
        img_vector = self.img_to_encoding(img)
        if img_vector is None:
            return None
        img_vector = [img_vector.flatten()]
        collection = db["vectors"]
        vectors = collection.find({})
        print(len(list(vectors)))
        img_vector = np.array(img_vector)
        for vector in vectors:
            if np.linalg.norm(np.array(vector['encoding']) - img_vector) < self.min_similarity:
                print("Hit in database")
                print("Time taken:", time() - start)
                print('')
                return vector['name']
        print("Miss in database")
        print("Time taken:", time() - start)
        print('')
        return None
        