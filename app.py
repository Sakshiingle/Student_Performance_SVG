from flask import Flask, request, render_template_string
import pickle
import numpy as np

# Set up the Flask application
app = Flask(__name__)

# Load the Support Vector Machine (SVC) model
try:
    with open('svm_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Error: 'svm_model.pkl' not found. Please check the filename.")

# --- Beautiful, Modern Aesthetic UI ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Prediction Model (SVM)</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
        
        body { 
            font-family: 'Inter', sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex; 
            justify-content: center; 
            align-items: center; 
            min-height: 100vh; 
            margin: 0;
            padding: 20px;
            color: #333;
        }
        
        /* Glassmorphism Card Effect */
        .card { 
            background: rgba(255, 255, 255, 0.9); 
            backdrop-filter: blur(10px);
            padding: 40px; 
            border-radius: 16px; 
            box-shadow: 0 15px 35px rgba(0,0,0,0.2); 
            width: 100%; 
            max-width: 650px; 
        }
        
        h2 { 
            text-align: center; 
            color: #2d3748; 
            margin-top: 0;
            font-weight: 600;
            letter-spacing: -0.5px;
            margin-bottom: 30px;
        }
        
        /* Grid Layout for 9 Features */
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        /* Make certain elements span full width if needed */
        .full-width {
            grid-column: 1 / -1;
        }
        
        .form-group { display: flex; flex-direction: column; }
        
        label { 
            font-weight: 600; 
            font-size: 13px;
            margin-bottom: 6px; 
            color: #4a5568; 
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        input, select { 
            width: 100%; 
            padding: 12px; 
            border: 1px solid #e2e8f0; 
            border-radius: 8px; 
            box-sizing: border-box; 
            font-size: 15px; 
            transition: all 0.3s ease;
            background-color: #f7fafc;
        }
        
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
            background-color: #fff;
        }
        
        button { 
            grid-column: 1 / -1;
            margin-top: 15px;
            padding: 16px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            border: none; 
            border-radius: 8px; 
            cursor: pointer; 
            font-size: 16px; 
            font-weight: 600; 
            letter-spacing: 1px;
            transition: transform 0.2s, box-shadow 0.2s; 
        }
        
        button:hover { 
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(118, 75, 162, 0.3);
        }
        
        .result-box { 
            margin-top: 30px; 
            padding: 20px; 
            text-align: center; 
            border-radius: 8px; 
            font-size: 18px; 
            font-weight: 600;
            animation: fadeIn 0.5s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .positive { background-color: #c6f6d5; color: #22543d; border: 1px solid #9ae6b4; }
        .negative { background-color: #fed7d7; color: #822727; border: 1px solid #feb2b2; }
        
        /* Mobile Responsiveness */
        @media (max-width: 600px) {
            .form-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="card">
        <h2>Student Outcome Predictor</h2>
        <form action="/predict" method="POST" class="form-grid">
            
            <div class="form-group">
                <label for="gender">Gender (0/1)</label>
                <input type="number" id="gender" name="gender" required placeholder="e.g. 1 for male , 0 for female" step="any">
            </div>
            
            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="age" required placeholder="e.g. 20" step="any">
            </div>
            
            <div class="form-group">
                <label for="study_hours_per_week">Study Hours / Week</label>
                <input type="number" id="study_hours_per_week" name="study_hours_per_week" required placeholder="e.g. 15" step="any">
            </div>
            
            <div class="form-group">
                <label for="attendance_rate">Attendance Rate</label>
                <input type="number" id="attendance_rate" name="attendance_rate" required placeholder="e.g. 85.5" step="any">
            </div>
            
            <div class="form-group">
                <label for="parent_education">Parent Education Level</label>
                <input type="number" id="parent_education" name="parent_education" required placeholder="e.g. 1, 2, or 3" step="any">
            </div>
            
            <div class="form-group">
                <label for="internet_access">Internet Access (0/1)</label>
                <input type="number" id="internet_access" name="internet_access" required placeholder="e.g. 1 for Yes" step="any">
            </div>
            
            <div class="form-group">
                <label for="extracurricular">Extracurriculars (0/1)</label>
                <input type="number" id="extracurricular" name="extracurricular" required placeholder="e.g. 1 for Yes" step="any">
            </div>
            
            <div class="form-group">
                <label for="previous_score">Previous Score</label>
                <input type="number" id="previous_score" name="previous_score" required placeholder="e.g. 75" step="any">
            </div>

            <div class="form-group full-width">
                <label for="final_score">Current/Baseline Final Score</label>
                <input type="number" id="final_score" name="final_score" required placeholder="e.g. 80" step="any">
            </div>
            
            <button type="submit">Run SVM Prediction</button>
        </form>

        {% if prediction_text %}
            <div class="result-box {% if prediction_value == 1 %}positive{% else %}negative{% endif %}">
                {{ prediction_text }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract all 9 features explicitly required by the SVC model
        features = [
            float(request.form.get('gender')),
            float(request.form.get('age')),
            float(request.form.get('study_hours_per_week')),
            float(request.form.get('attendance_rate')),
            float(request.form.get('parent_education')),
            float(request.form.get('internet_access')),
            float(request.form.get('extracurricular')),
            float(request.form.get('previous_score')),
            float(request.form.get('final_score'))
        ]
        
        # Format for scikit-learn model
        final_features = np.array([features])
        
        # Predict
        prediction = model.predict(final_features)
        output = int(prediction[0])
        
        # Determine display text based on Class 0 or 1
        if output == 1:
            result_text = "Prediction: Class 1 (Positive Outcome)"
        else:
            result_text = "Prediction: Class 0 (Alternative Outcome)"
            
        return render_template_string(HTML_TEMPLATE, prediction_text=result_text, prediction_value=output)
        
    except Exception as e:
        return f"<div style='color:red; text-align:center; margin-top:20px;'>An error occurred: {str(e)}</div>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
