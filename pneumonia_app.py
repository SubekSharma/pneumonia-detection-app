import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('model.h5')

def preprocess_image(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255
    img = np.expand_dims(img, axis=0)
    return img

def get_class_label(predictions):
    class_index = 1 if predictions > 0.5 else 0
    class_labels = ["Normal Chest", "Pneumonic Chest"]
    return class_labels[class_index]

def main():
    st.set_page_config(page_title="Pneumonia Prediction", page_icon="👨‍⚕️", layout="centered")

    st.title("Pneumonia Prediction")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = preprocess_image(uploaded_file)
        predictions = model.predict(image)
        class_label = get_class_label(predictions)

        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("Prediction:", class_label)

if __name__ == "__main__":
    main()
