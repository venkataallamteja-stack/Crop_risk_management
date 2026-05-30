from flask import Flask, request, render_template
import joblib
import numpy as np


app = Flask(__name__)

# Load the trained AI model
model = joblib.load('crop_model.pkl')

# Risk management function
def get_management_plan(n, p, k, ph, rain):

    risks = []
    advice = []

    # Soil pH analysis
    if ph < 5.0:
        risks.append("High Soil Acidity")
        advice.append("Apply lime to neutralize soil pH.")

    elif ph > 8.5:
        risks.append("High Soil Alkalinity")
        advice.append("Add gypsum to lower soil pH.")

    # Rainfall analysis
    if rain < 50:
        risks.append("Severe Drought Risk")
        advice.append("Increase irrigation frequency.")

    elif rain > 350:
        risks.append("Flood Risk")
        advice.append("Improve field drainage.")

    # Nitrogen analysis
    if n < 15:
        risks.append("Nitrogen Deficiency")
        advice.append("Apply nitrogen fertilizers.")

    # Final risk result
    if risks:
        risk_result = " & ".join(risks)
    else:
        risk_result = "Low Risk (Optimal Conditions)"

    # Final management advice
    if advice:
        plan_result = " ".join(advice)
    else:
        plan_result = "No special management action required."

    return risk_result, plan_result


# Home page route
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Get form values
        input_values = [float(x) for x in request.form.values()]

        # Convert input into numpy array
        final_input = [np.array(input_values)]

        # Crop prediction
        prediction = model.predict(final_input)

        crop_name = prediction[0].capitalize()

        # Risk analysis
        risk_lvl, strategy = get_management_plan(
            input_values[0],  # Nitrogen
            input_values[1],  # Phosphorus
            input_values[2],  # Potassium
            input_values[5],  # pH
            input_values[6]   # Rainfall
        )

        return render_template(
            'index.html',
            prediction_text=f"Recommended Crop: {crop_name}",
            risk_text=f"Risk Level: {risk_lvl}",
            mgmt_text=f"Management Strategy: {strategy}"
        )

    except Exception as e:

        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}"
        )


# Run Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)