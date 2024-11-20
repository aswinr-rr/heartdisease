import numpy as np
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the model using joblib
model = joblib.load('D:/ML_Proj/decision_tree_model.pkl')

# Function to convert age to a numeric AgeCategory
def age_to_category(age):
    """Converts a user's age input to a numeric 'AgeCategory'."""
    if age < 18:
        return 1  # Assuming age <18 will default to '18-24' (encoded as 1)
    elif 18 <= age < 25:
        return 1
    elif 25 <= age < 30:
        return 2
    elif 30 <= age < 35:
        return 3
    elif 35 <= age < 40:
        return 4
    elif 40 <= age < 45:
        return 5
    elif 45 <= age < 50:
        return 6
    elif 50 <= age < 55:
        return 7
    elif 55 <= age < 60:
        return 8
    elif 60 <= age < 65:
        return 9
    elif 65 <= age < 70:
        return 10
    elif 70 <= age < 75:
        return 11
    elif 75 <= age < 80:
        return 12
    else:
        return 13  # For '80 or older'

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Collect age input from form and convert it to numeric AgeCategory
    age = float(request.form.get('Age'))
    age_category = age_to_category(age)
    
    # Collect the rest of the features from the form, using 0 for Race and GenHealth
    input_data = [
        float(request.form.get('BMI')),
        float(request.form.get('Smoking')),
        float(request.form.get('AlcoholDrinking')),
        float(request.form.get('Stroke')),
        float(request.form.get('PhysicalHealth')),
        float(request.form.get('MentalHealth')),
        float(request.form.get('DiffWalking')),
        float(request.form.get('Sex')),
        age_category,  # Use the converted age category here
        0.0,  # Default value for Race
        float(request.form.get('Diabetic')),
        float(request.form.get('PhysicalActivity')),
        0.0,  # Default value for GenHealth
        float(request.form.get('SleepTime')),
        float(request.form.get('Asthma')),
        float(request.form.get('KidneyDisease')),
        float(request.form.get('SkinCancer'))
    ]
    
    # Convert input data to numpy array and ensure it’s in float format
    data_array = np.array([input_data], dtype=float)
    
    # # Make prediction
    # prediction = model.predict(data_array)
    # result = "Positive" if prediction[0] ==1 else "Negative"

    # Get probability prediction
    probability = model.predict_proba(data_array)[0][1]  # Probability for class 1
    threshold = 0.3  # Adjust if necessary
    result = "Positive" if probability > threshold else "Negative"
    
    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True)
