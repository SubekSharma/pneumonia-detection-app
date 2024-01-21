import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5') 
# Define the target size for the model
img_size = (224, 224)

# Function to preprocess the image
def preprocess_image(img):
    img = image.load_img(img, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict_image(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predictions = np.squeeze(predictions, axis=0)

    return predictions

# Streamlit app
def main():
    st.title("Pneumonia Detection")

    uploaded_file = st.file_uploader("Upload an image...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        
        # Make predictions
        prediction = predict_image(uploaded_file)

        # Display the results
        st.write("**Prediction:**")
        if prediction > 0.8:
            st.write("The image is classified as **Pneumonia**.")
        else:
            st.write("The image is classified as **Normal**.")

        # confidence = prediction if prediction > 0.5 else 1 - prediction
        # st.write("**Confidence:**")
        # st.write(f"{confidence*100:.2f}%")

if __name__ == "__main__":
    main()
