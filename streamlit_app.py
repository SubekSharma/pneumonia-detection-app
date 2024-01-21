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
    prediction = model.predict(img_array)[0][0]  # Single output value

    return prediction

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
        if prediction > 0.5:
            st.write("The image is classified as **Pneumonia**.")
        else:
            st.write("The image is classified as **Normal**.")

        # Display the confidence directly
        st.write("**Confidence:**")
        st.write(f"{prediction*100:.2f}%")

# Example instructions
st.markdown("""
    Example Instructions:
    - Upload a chest X-ray image in JPG format.
    - The app will predict whether the image is normal or pneumonia.
""")

if __name__ == "__main__":
    main()


