"""
Routes and views for the flask application.
"""

import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from flask import render_template, jsonify, request, send_file
from datetime import datetime
import os
import urllib.request
from DiabeticPrediction import app


@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'layout.html',
        title='Diabetic Prediction based on Patient BMI',
        year=datetime.now().year,
    )


@app.route('/api/diabeticprediction', methods=['POST'])
def detect_diabetic():
    try:
        data = request.get_json()
        pname = data['pname']
        gender = int(data['gender'])
        age = int(data['age'])
        hypertension = int(data['hypertension'])
        heart_disease = int(data['heartDisease'])
        smoking = int(data['smoking'])
        bmi = float(data['bmi'])
        hemoglobin = float(data['hemoglobin'])
        glucose = int(data['glucose'])
        gender_mapping = {
            'Female': 0,
            'Male': 1,
            'Other': 2
        }

        smoking_mapping = {
            'never': 0,
            'ever': 1,
            'former': 2,
            'No Info': 3,
            'current': 4,
            'not current': 5
        }

        data_values = [[gender, age, hypertension, heart_disease, smoking, bmi, hemoglobin, glucose]]
        print(data_values)
        
        path = os.getcwd()
        filepath = path + "/Diabetes_DB.csv"
        # define header names
        header_names = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level',
                        'blood_glucose_level', 'diabetes']
        # display table
        print(data_values)
        df = pd.read_csv(filepath, names=header_names)
        # Replace values in the "gender & smoking" column
        df['gender'] = df['gender'].replace(gender_mapping)
        df['smoking_history'] = df['smoking_history'].replace(smoking_mapping)
        df.head()
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        classifier = RandomForestClassifier(n_estimators=100)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        # print(y_pred)
        result = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(result)
        result1 = classification_report(y_test, y_pred)
        print("Classification Report:", )
        print(result1)
        result2 = accuracy_score(y_test, y_pred)
        print("Accuracy:", result2)

        x_test = data_values
        y_pred = classifier.predict(x_test)
        print(y_pred)
        return jsonify({'data': pname + " is diabetic" if y_pred[0] == 1 else pname + " is not diabetic"})
    except Exception as err:
        return jsonify({'data': err})
