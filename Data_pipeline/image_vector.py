import tensorflow as tf
import numpy as np

def img_to_encoding(img_array):
    gpus = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available", len(gpus))
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    model = tf.keras.models.load_model('../model')  # Load the model
    # Ensure the input image is in the correct shape (160, 160, 3)
    img = np.around(img_array / 255.0, decimals=12)  # Normalize the image array
    img = np.resize(img, (160, 160, 3))  # Resize the image to (160, 160, 3) if needed
    x_train = np.expand_dims(img, axis=0)  # Add a dimension of 1 as the first dimension
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)
