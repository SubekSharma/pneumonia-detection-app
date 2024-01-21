import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

# Load pretrained model
model_path = "model.h5"
model = tf.keras.models.load_model(model_path)

# Define the class labels
class_labels = {0: 'Normal', 1: 'Pneumonia'}

# Create a Streamlit app
st.title("Pneumonia Classification App")

# Allow user to upload an image
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type="jpg")

if uploaded_file is not None:
    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Display the result
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Class:", class_labels[predicted_class])
    st.write("Confidence:", prediction[0][predicted_class])

# Example instructions
st.markdown("""
    Example Instructions:
    - Upload a chest X-ray image in JPG format.
    - The app will predict whether the image is normal or pneumonia.
""")
