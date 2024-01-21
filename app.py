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
    html_content = """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pneumonia Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
        }

        .container {
            text-align: center;
            padding: 20px;
            border: 1px solid #ccc;
            border-radius: 8px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }

        input[type="file"] {
            margin: 10px 0;
        }

        button {
            padding: 10px;
            background-color: #007bff;
            color: #fff;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }

        button:hover {
            background-color: #0056b3;
        }

        #predictionResult {
            margin-top: 20px;
            font-weight: bold;
        }
    </style>
    </style>
</head>
<body>
    <div class="container">
        <h1>Pneumonia Prediction</h1>
        <input type="file" id="imageInput" accept="image/*">
        <button onclick="performPrediction()">Predict</button>
        <div id="predictionResult"></div>
    </div>

    <script>
        function performPrediction() {
            const imageInput = document.getElementById('imageInput');
            const predictionResult = document.getElementById('predictionResult');
        
            const file = imageInput.files[0];
            if (file) {
                const formData = new FormData();
                formData.append('file', file);
        
                const predictionEndpoint = '/predict';
        
                fetch(predictionEndpoint, {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    predictionResult.innerHTML = `Prediction: ${data.prediction}`;
                })
                .catch(error => {
                    console.error('Error:', error);
                    predictionResult.innerHTML = 'Error occurred during prediction.';
                });
            } else {
                predictionResult.innerHTML = 'Please select an image.';
            }
        }
        
    </script>
</body>
</html>
    """
    return render_template_string(html_content)

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
