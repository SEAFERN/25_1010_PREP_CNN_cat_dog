import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Usage: python predict.py path_to_image.jpg
if len(sys.argv) != 2:
    print("Usage: python predict.py path_to_image.jpg")
    sys.exit(1)

img_path = sys.argv[1]

# Load the trained model
model = tf.keras.models.load_model('cat_dog_cnn_model.h5')

# Image parameters (should match training)
img_height, img_width = 150, 150

# Load and preprocess the image
img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make prediction
prediction = model.predict(img_array)[0][0]

if prediction > 0.5:
    print(f"Prediction: Dog ({prediction:.2f})")
else:
    print(f"Prediction: Cat ({1-prediction:.2f})")
