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
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict_image(img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    prediction = np.squeeze(prediction, axis=0)
    return prediction

# Streamlit app
def main():
    st.title("Pneumonia Detection")

    uploaded_file = st.file_uploader("Upload an image...", type="jpg")

    # Example instructions
    st.markdown("""
        Example Instructions:
        - Upload a chest X-ray image in JPG format.
        - The app will predict whether the image is normal or pneumoniac.
    """)

    # Provide links to download sample images
    st.write("**Download Sample Images:**")
    pneumonic_download = st.button("Download Pneumonic Image")
    normal_download = st.button("Download Normal Image")

    if pneumonic_download:
        urllib.request.urlretrieve("https://prod-images-static.radiopaedia.org/images/8589259/debc366fbee881069b1bd4b23a8020_big_gallery.jpg", "pneumonic_image.jpg")
        st.success("Pneumonic image downloaded successfully!")

    if normal_download:
        urllib.request.urlretrieve("https://www.kenhub.com/thumbor/AkXFsw0396y894sLEMWlcDuChJA=/fit-in/800x1600/filters:watermark(/images/logo_url.png,-10,-10,0):background_color(FFFFFF):format(jpeg)/images/library/10851/eXtmE1V2XgsjZK2JolVQ5g_Border_of_left_atrium.png", "normal_image.jpg")
        st.success("Normal image downloaded successfully!")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        
        # Make predictions
        prediction = predict_image(uploaded_file)

        # Display the results
        st.write("**Prediction:**")
        class_label = "Pneumonia" if prediction > 0.5 else "Normal"
        st.write(f"The image is classified as **{class_label}**.")



if __name__ == "__main__":
    main()
