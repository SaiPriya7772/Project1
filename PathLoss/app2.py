from flask import *
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the saved Random Forest model and scaler
model = joblib.load('rf_path_loss_model.pkl')
scaler = joblib.load('scaler.pkl')


@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Expecting a JSON request with the following input structure
    try:
        input_string = request.form['features']
        input_list = input_string.split(',')


        features = np.array([float(x) for x in input_list[:]]).reshape(1,-1)
        # Scale the input features
        features_scaled = scaler.transform(features)

        # Make the prediction using the model
        prediction = model.predict(features_scaled)

        # Return the prediction as a JSON response
        return render_template('index.html', prediction_text=f'Predicted Path Loss: {prediction[0]:.2f} dB')

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
