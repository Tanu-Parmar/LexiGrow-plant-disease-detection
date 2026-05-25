import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("../trained_models/lexigrow_cnn.h5")
IMG_SIZE = 128

def predict(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    return np.argmax(prediction), prediction
