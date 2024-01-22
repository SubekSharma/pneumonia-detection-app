import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import base64

hide_streamlit_style = """
                <style>
                [data-testid="stToolbar"] {visibility: hidden !important;}
                footer {visibility: hidden !important;}
                
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#Load the pre-trained model
model = tf.keras.models.load_model('model.h5') 
img_size = (224, 224)

def preprocess_image(img):
    img = image.load_img(img, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    prediction = np.squeeze(prediction, axis=0)
    return prediction

def display_image_with_download(image_path, caption, download_text):
    image = Image.open(image_path)
    st.image(image, caption=caption, use_column_width=True)
    
    """Generate a download link"""
    with open(image_path, 'rb') as f:
        data = f.read()
        base64_data = base64.b64encode(data).decode('utf-8')
        href = f'<a href="data:application/octet-stream;base64,{base64_data}" download="{download_text}.jpg">Download {download_text}</a>'
        st.markdown(href, unsafe_allow_html=True)



def main():
    st.title("Pneumonia Detection")

    uploaded_file = st.file_uploader("Upload a chest X-ray image in JPG format...", type="jpg")

    st.markdown("""
        Example Instructions:
        - Upload a chest X-ray image in JPG format.
        - Or, download sample images below and check the predictions.
    """)

    st.write("**Download Sample Images:**")
    
    pneumonic_download = st.button("Download Pneumonic Image")
    normal_download = st.button("Download Normal Image")

    if pneumonic_download:
        pneumonic_image_path = "test-pneumonia_028.jpg"  # Replace with actual path
        display_image_with_download(pneumonic_image_path, "Pneumonic Image", "Pneumonic Image")

    if normal_download:
        normal_image_path = "test-normal_001.jpg"  # Replace with actual path
        display_image_with_download(normal_image_path, "Normal Image", "Normal Image")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        prediction = predict_image(uploaded_file)

        st.write("**Prediction:**")
        class_label = "Pneumonia" if prediction > 0.5 else "Normal"
        st.write(f"The image is classified as **{class_label}**.")    


if __name__ == "__main__":
    main()
