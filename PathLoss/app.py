from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


# Initialize Flask app
app = Flask(__name__)

# Load the saved XGBoost model
with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

lgb_model = joblib.load('lightgbm_model.pkl')


s=['FallL', 'FallH', 'FallM', 'SpringL', 'SpringH', 'SpringM',
       'SummerL', 'SummerH', 'SummerM', 'WinterL', 'WinterH', 'WinterM',
       'winterL', 'winterM', 'winterH']
label_encoder = LabelEncoder()
label_encoder.fit(s)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input string from the form
    input_string = request.form['features']  # Example: 'FallL,2,417.2,...'

    # Split the string by commas to get the individual values
    input_list = input_string.split(',')

    # Extract the categorical variable (season) and convert it using LabelEncoder
    season = input_list[0]  # Example: 'FallL'
    
    # Since 'FallL' contains a typo, correct it by trimming excess characters
    season = season.strip().replace('L', '')

    season_encoded = label_encoder.transform([season])[0]  # Convert to numeric

    # Convert the rest of the inputs to float
    numeric_features = [float(x) for x in input_list[1:]]

    # Combine the encoded categorical variable with numeric features
    final_features = [season_encoded] + numeric_features
    final_features = np.array(final_features).reshape(1, -1)


    
    # Make prediction using the model
    prediction = lgb_model.predict(final_features)
    
    # Return the result to the frontend
    return render_template('index.html', prediction_text=f'Predicted Path Loss: {prediction[0]:.2f} dB')

if __name__ == "__main__":
    app.run(debug=True)
