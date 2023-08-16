from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from tensorflow.keras.models import model_from_json
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Load the model architecture from the JSON file
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)

# Load the model weights from the H5 file
model.load_weights('model.h5')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    image_file = request.files['image']
    image_filename = secure_filename(image_file.filename)

    # Save the image to the 'static/uploads' directory
    image_path = os.path.join(app.root_path, 'static/uploads', image_filename)
    image_file.save(image_path)

    # Open the image using PIL
    image = Image.open(image_file)

    # Preprocess the image
    img = image.resize((128, 128))
    img = np.array(img.convert('L'))
    img = img.reshape(1, 128, 128, 1)
    img = img / 255.0

    # Make predictions
    gender_pred, age_pred = model.predict(img)

    # Process the predictions
    gender_pred = 'Female' if gender_pred > 0.5 else 'Male'
    age_pred = int(age_pred[0][0])


    # Render the result template with the predicted age, gender, ethnicity, and image filename
    return render_template('result.html', gender=gender_pred, age=age_pred, image_filename=image_filename)

if __name__ == '__main__':
    app.run(debug=True)