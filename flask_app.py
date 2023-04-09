import os
import io
import requests
from flask import Flask, request, jsonify,render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras

#set openapi key
OPENAI_API_KEY = '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'
# Load the dog breed labels
def get_unique_breeds(path):
    label_csv = pd.read_csv(path)
    labels = np.array(label_csv['breed'])
    unique_labels = np.unique(labels)
    return unique_labels
dog_breeds = get_unique_breeds('C:/Users/ACER NITRO/computer vision/computer vision udemy/dogify/labels.csv')

# Load the TensorFlow model
model_path = 'models/20230330-12441680180287full-image-set-imagenetv3-Adam-lr-0.0002final.h5'
model = tf.keras.models.load_model(model_path)

# Initialize the Flask app
app = Flask(__name__)

# Define a route to the predict function
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if an image was submitted
        if 'image' not in request.files:
            return jsonify({'error': 'no image found'})

        # Read the image file
        image_file = request.files['image']
        image_bytes = io.BytesIO(image_file.read())

        # Preprocess the image and make a prediction
        image = Image.open(image_bytes)
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)[0]

        # Get the predicted dog breed and confidence score
        dog_breed = dog_breeds[np.argmax(prediction)]
        confidence_score = round(prediction[np.argmax(prediction)], 2)

        # Use OpenAI to explain the dog breed
        prompt = f"Explain {dog_breed} in simple terms"
        data = {"prompt": prompt, "temperature": 0.5, "max_tokens": 100}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ[OPENAI_API_KEY]}",
        }
        response = requests.post("https://api.openai.com/v1/engines/davinci-codex/completions", headers=headers, json=data)
        explanation = response.json()["choices"][0]["text"].strip()

        # Return the prediction and explanation as JSON
        return jsonify({'breed': dog_breed, 'confidence': confidence_score, 'explanation': explanation})
    else:
        return home()

@app.route('/')
def home():
    return render_template('predict.html')
# Start the Flask app
if __name__ == '__main__':
    app.run()
