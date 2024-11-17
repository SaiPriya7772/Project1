from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Load the saved neural network model
model = tf.keras.models.load_model('neural_network_model.h5')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get features from the form and process input string
    input_str = request.form['features']
    features = [float(x) for x in input_str.split(',')]
    final_features = np.array(features).reshape(1, -1)

    # Make prediction using the model
    prediction = model.predict(final_features)
    return render_template('index.html', prediction_text=f'Predicted Path Loss: {prediction[0][0]:.2f} dB')

if __name__ == "__main__":
    app.run(debug=True)
