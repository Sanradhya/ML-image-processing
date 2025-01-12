from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# Function to process image using PIL
def process_image_with_pil(image_path):
    """
    Load an image, convert to grayscale, and invert it.
    """
    image = Image.open(image_path)
    gray_image = image.convert('L')  # Convert to grayscale
    gray_image.show()
    image_array = np.array(gray_image)
    inverted_image = 255 - image_array
    inverted_image_pil = Image.fromarray(inverted_image)
    inverted_image_pil.show()
    return inverted_image_pil

# Function to use a pre-trained ML model for inference
def use_machine_learning_model(image_path):
    """
    Load the pre-trained model and predict the image's class.
    """
    model = tf.keras.models.load_model('model/model.h5')  # Path to pre-trained model
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize to model's input size
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    predictions = model.predict(image)
    return predictions
