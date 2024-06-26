from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import pickle
from flask_session import Session

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Load the machine learning model
model = pickle.load(open(r'C:\sudhanshu_projects\project-task-training-course\student_marks_prediction.pkl', 'rb'))

# Initialize an empty dataframe to store user credentials and study hours/predicted values
user_credentials = pd.DataFrame(columns=['username', 'password'])
df = pd.DataFrame()

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def handle_login():
    username = request.form['username']
    password = request.form['password']
    global user_credentials
    
    # Create a new DataFrame row
    new_user = pd.DataFrame([[username, password]], columns=['username', 'password'])
    # Append the new row to the DataFrame
    user_credentials = pd.concat([user_credentials, new_user], ignore_index=True)
    
    # Save the user credentials to a CSV file
    user_credentials.to_csv(r'C:\sudhanshu_projects\project-task-training-course\student-marks-prediction\user_credentials_sample.csv', index=False)
    
    # Store the username in the session
    session['username'] = username
    
    return redirect(url_for('index'))

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global df
    
    study_hours = float(request.form['study_hours'])
    prediction = model.predict([[study_hours]])[0][0]
    
    if prediction>100:
        prediction = 100
    
    # Retrieve the username from the session
    username = session.get('username')
    
    # Append the input, predicted value, and username to df then save in CSV file
    new_data = pd.DataFrame({'Username': [username], 'Study Hours': [study_hours], 'Predicted Output': [prediction]})
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(r'C:\sudhanshu_projects\project-task-training-course\student-marks-prediction\smp_data_from_app.csv', index=False)
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
