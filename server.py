from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model (make sure to provide the correct path to your model)
model = load_model('final_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    
    # Parse and preprocess the data
    # Assuming the data is a list of lists, each inner list representing the features for one sample
    features = np.array(data['features'])
    # Normalize/scale the feature data
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Make predictions using the model
    predictions = model.predict(features)
    
    # Convert predictions to class indices
    predicted_classes = np.argmax(predictions, axis=2).tolist()
    
    # Return the predictions as a JSON response
    return jsonify({'predictions': predicted_classes})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
