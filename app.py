from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

model = tf.keras.models.load_model('model.h5')

def preprocess_image(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255
    img = np.expand_dims(img, axis=0)
    return img

def get_class_label(predictions):
    class_index = 1 if predictions>0.5 else 0
    class_labels = ["Normal Chest", "Pneumonic Chest"]
    return class_labels[class_index]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file selected'})

    file = request.files['file']
    img_array = preprocess_image(file)
    predictions = model.predict(img_array)
    class_label = get_class_label(predictions)

    return jsonify({'prediction': class_label})

# if __name__ == '__main__':
#     app.run(debug=True)
