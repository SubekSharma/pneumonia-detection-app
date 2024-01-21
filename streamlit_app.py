import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')  # Update with your actual model file

# Define the target size for the model
img_size = (224, 224)

# Function to preprocess the image
def preprocess_image(img):
    img = image.load_img(img, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to make predictions
def predict_image(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    return predictions

# Streamlit app
def main():
    st.title("Pneumonia Classification App")
    st.sidebar.title("Upload Image")

    uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.sidebar.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.sidebar.write("")
        st.sidebar.write("Classifying...")

        # Make predictions
        predictions = predict_image(uploaded_file)

        # Display the results
        st.write("**Prediction:**")
        if predictions[0][0] > 0.5:
            st.write("The image is classified as **Pneumonia**.")
        else:
            st.write("The image is classified as **Normal**.")

        st.write("**Confidence:**")
        st.write(f"Pneumonia: {predictions[0][0] * 100:.2f}%")
        st.write(f"Normal: {predictions[0][1] * 100:.2f}%")

if __name__ == "__main__":
    main()
