import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from flask import Flask, request, render_template, jsonify

#initialize Flask app
app = Flask(__name__)

#load dataset
try:
    insurance_data = pd.read_csv("insurance.csv")
except FileNotFoundError:
    insurance_data = pd.DataFrame({
        "age": [], "sex": [], "bmi": [], "children": [], "smoker": [],
        "region": [], "charges": []
    })

def preprocess_data(data):
    data = data.copy()
    data["sex"] = data["sex"].map({"male": 0, "female": 1})
    data["smoker"] = data["smoker"].map({"no": 0, "yes": 1})
    
    #being sure numerical fields are of the correct type
    data["age"] = pd.to_numeric(data["age"], errors='coerce')
    data["bmi"] = pd.to_numeric(data["bmi"], errors='coerce')
    data["children"] = pd.to_numeric(data["children"], errors='coerce')
    
    if "region" in data.columns:
        data = pd.get_dummies(data, columns=["region"], drop_first=True)
    
    return data


processed_data = preprocess_data(insurance_data)

# train a regression model for medical cost prediction
def train_cost_model(data):
    if data.empty:
        return None, None, None
    #######FEATURES : age,bmi,children,smoker
    features = data[["age", "bmi", "children", "smoker"]]
    target = data["charges"]  # Use charges as the target
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate_cost_model(model, X_test, y_test):#calculate accurancy
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print("Linear Regression Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared Score: {r2:.2f}")

#training the cost model
cost_model, X_cost_test, y_cost_test = train_cost_model(processed_data)
if cost_model:
    evaluate_cost_model(cost_model, X_cost_test, y_cost_test)


#traiming a classification model for health recommendations (bmi and smoking)
def train_health_model(data):
    if data.empty:
        return None, None, None
    #######FEATURES : age,bmi,children,smoker
    features = data[["age", "bmi", "children", "smoker"]]
    target = data["charges"]
    target = target > target.median()  # Binary classification: high or low health risk
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_health_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)
    print("Logistic Regression Evaluation Metrics:")
    print(f"Accuracy: {acc:.2f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

health_model, X_health_test, y_health_test = train_health_model(processed_data)
if health_model:
    evaluate_health_model(health_model, X_health_test, y_health_test)

#detailed recommendations based on user data
def generate_detailed_recommendation(user_data):
    age = user_data["age"].values[0]
    bmi = user_data["bmi"].values[0]
    smoker = user_data["smoker"].values[0]
    
    recommendation = ""

    #recommendations on bmi based
    if bmi < 18.5:
        recommendation += "Your BMI is below the healthy range (underweight). It's important to focus on gaining healthy weight through a balanced diet. Consult with a healthcare provider for personalized advice.\n"
    elif 18.5 <= bmi < 24.9:
        recommendation += "Your BMI is within the normal weight range. Keep up the good work by maintaining a healthy diet and regular physical activity.\n"
    elif 25 <= bmi < 29.9:
        recommendation += "Your BMI is in the overweight range. It would be beneficial to work on losing weight through regular exercise and a balanced diet.\n"
    else:
        recommendation += "Your BMI is in the obese range. It's important to focus on weight loss through a healthy diet, regular physical activity, and possibly consult with a healthcare provider.\n"

    #recommendations ob smoking based 
    if smoker == 1:
        recommendation += "As a smoker, you are at a higher risk for many health conditions. Quitting smoking will have immediate and long-term benefits for your health. We strongly recommend seeking support to quit.\n"
    else:
        recommendation += "Great job! Staying smoke-free is one of the best things you can do for your long-term health.\n"

    #recommendations on age based 
    if age < 30:
        recommendation += "As a young adult, now is the best time to build healthy habits that will benefit you in the long term. Stay active, eat nutritious food, and avoid smoking.\n"
    elif 30 <= age < 50:
        recommendation += "As you are entering your 30s, it's important to keep an eye on your health. Regular check-ups and maintaining a balanced lifestyle will keep you in good health.\n"
    else:
        recommendation += "As you age, it's essential to focus on maintaining mobility, flexibility, and cardiovascular health. Regular exercise, a balanced diet, and routine health screenings are key.\n"
    
    return recommendation

#Flask routes
@app.route("/")
def home():
    return render_template("index.html")  #serving the HTML form

@app.route("/predict_charges", methods=["POST"])
def predict_charges():
    data = request.form.to_dict()  #fet the form data
    user_data = pd.DataFrame([data])
    user_data = preprocess_data(user_data)
    
    #being sure about the model is defined
    if cost_model is None:
        return "Error: The cost prediction model has not been trained properly."
    
    #predict charges using the cost model
    # Modelden tahmini al
    predicted_cost = cost_model.predict(user_data[["age", "bmi", "children", "smoker"]])[0]

    # 1000'den küçükse 1000 yap
    charges = 1000 if predicted_cost < 1000 else predicted_cost    #start charge point via 1000

    return render_template("result.html", result="{:,.2f}".format(charges))  #show result in a new page

@app.route("/health_recommendation", methods=["POST"])
def health_recommendation():
    data = request.form.to_dict()
    #print("Received Data:", data)  #debugging print
    user_data = pd.DataFrame([data])
    
    #check if sex exists in the data
    if 'sex' not in user_data.columns:
        return "Error: Missing 'sex' field in the data."

    user_data = preprocess_data(user_data)
    
    #predict health risk based on users data
    risk = health_model.predict(user_data[["age", "bmi", "children", "smoker"]])[0]
    
    #generate detailed recommendations
    recommendation = generate_detailed_recommendation(user_data)
    
    #define health recommendations based on risk category
    recommendations = {
        0: "You are in the low-risk category. Keep maintaining your current healthy lifestyle!",
        1: "You are in the high-risk category. We recommend improving your BMI through exercise, a balanced diet, and quitting smoking if applicable."
    }
    
    return render_template("recommendation.html", recommendation=recommendation)

if __name__ == "__main__":
    app.run(debug=True)