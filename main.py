from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('deploy_test.pkl')

@app.route('/')
def Welcome():
    return render_template('base.html')

@app.route('/predict', methods =['post'])
def predict():
    experience = request.form.get('experience')
    test_score = request.form.get('test_score')
    interview_score = request.form.get('interview_score')
    
    prediction = model.predict([[experience, test_score, interview_score]])

    return render_template('base.html', predicted_value = f'Employee salary will be ${round(prediction[0],2)}')



app.run(debug=True)