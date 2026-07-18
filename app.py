import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained Gaussian Naive Bayes model
MODEL_PATH = "naive_model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None

# Single-file HTML template with modern CSS embedded directly inside it
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Naive Bayes Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            --card-bg: #ffffff;
            --text-main: #2d3748;
            --text-muted: #718096;
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --border-color: #e2e8f0;
            --success: #10b981;
            --shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
        }

        .container {
            background: var(--card-bg);
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: var(--shadow);
            width: 100%;
            max-width: 500px;
            transition: transform 0.2s;
        }

        h2 {
            font-weight: 700;
            color: #1a202c;
            margin-bottom: 0.5rem;
            text-align: center;
        }

        .subtitle {
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-bottom: 2rem;
            text-align: center;
        }

        .error-banner {
            background-color: #fee2e2;
            color: #991b1b;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            font-size: 0.9rem;
            text-align: center;
            border: 1px solid #fca5a5;
        }

        .form-group {
            margin-bottom: 1.25rem;
        }

        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
            font-size: 0.9rem;
            color: #4a5568;
        }

        input[type="number"], select {
            width: 100%;
            padding: 0.75rem 1rem;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-family: inherit;
            font-size: 1rem;
            color: var(--text-main);
            background-color: #f8fafc;
            transition: all 0.2s;
        }

        input[type="number"]:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
            background-color: #fff;
        }

        button {
            width: 100%;
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 0.85rem;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s, transform 0.1s;
            margin-top: 1rem;
            box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2);
        }

        button:hover {
            background-color: var(--primary-hover);
        }

        button:active {
            transform: scale(0.98);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.5rem;
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 12px;
            text-align: center;
            animation: fadeIn 0.4s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #166534;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }

        .result-value {
            font-size: 2rem;
            font-weight: 800;
            color: #14532d;
            margin-bottom: 0.5rem;
        }

        .confidence {
            font-size: 0.85rem;
            color: #15803d;
        }
    </style>
</head>
<body>

<div class="container">
    <h2>Model Deployment Dashboard</h2>
    <div class="subtitle">Gaussian Naive Bayes Predictor</div>

    {% if error %}
        <div class="error-banner">{{ error }}</div>
    {% endif %}

    <form method="POST" action="/">
        <div class="form-group">
            <label for="gender">Gender</label>
            <select id="gender" name="Gender" required>
                <option value="1" {% if inputs and inputs['Gender'] == '1' %}selected{% endif %}>Male</option>
                <option value="0" {% if inputs and inputs['Gender'] == '0' %}selected{% endif %}>Female</option>
            </select>
        </div>

        <div class="form-group">
            <label for="age">Age</label>
            <input type="number" id="age" name="Age" placeholder="e.g. 35" min="0" max="120" 
                   value="{{ inputs['Age'] if inputs else '' }}" required>
        </div>

        <div class="form-group">
            <label for="salary">Estimated Salary</label>
            <input type="number" id="salary" name="EstimatedSalary" placeholder="e.g. 50000" min="0" 
                   value="{{ inputs['EstimatedSalary'] if inputs else '' }}" required>
        </div>

        <button type="submit">Generate Prediction</button>
    </form>

    {% if prediction is not none %}
        <div class="result-box">
            <div class="result-title">Prediction Class Result</div>
            <div class="result-value">{{ prediction }}</div>
            {% if confidence %}
                <div class="confidence">Confidence Score: <strong>{{ confidence }}%</strong></div>
            {% endif %}
        </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if not model:
        return render_template_string(HTML_TEMPLATE, error="Error: 'naive_model.pkl' was not found on the server cluster.", prediction=None)

    prediction = None
    confidence = None
    inputs = None

    if request.method == "POST":
        try:
            # Extract inputs directly from form elements
            inputs = {
                "Gender": request.form.get("Gender"),
                "Age": request.form.get("Age"),
                "EstimatedSalary": request.form.get("EstimatedSalary")
            }
            
            # Convert values to correct numerical types
            gender = float(inputs["Gender"])
            age = float(inputs["Age"])
            salary = float(inputs["EstimatedSalary"])
            
            # Shape the arrays correctly for the pipeline
            features = np.array([[gender, age, salary]])
            
            # Execute standard predictions using scikit-learn
            pred_class = model.predict(features)[0]
            pred_proba = model.predict_proba(features)[0]
            
            # Set values to return to our UI display
            prediction = int(pred_class)
            confidence = round(float(pred_proba[prediction]) * 100, 2)

        except Exception as e:
            return render_template_string(HTML_TEMPLATE, error=f"Processing Error: {str(e)}", prediction=None, inputs=inputs)

    return render_template_string(HTML_TEMPLATE, prediction=prediction, confidence=confidence, inputs=inputs, error=None)

if __name__ == "__main__":
    # Binds dynamically to the runtime environment port specified by Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
