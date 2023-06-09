from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__, static_folder='images')

model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        age = request.form['age'] 
        sex = request.form['sex'] 
        cp = request.form['cp']
        trestbps = request.form['trestbps']
        chol = request.form['chol']
        fbs = request.form['fbs']
        restecg = request.form['restecg']
        thalach = request.form['thalach']
        exang = request.form['exang']
        oldpeak = request.form['oldpeak']
        slope = request.form['slope']
        ca = request.form['ca']
        thal = request.form['thal']
        arr = np.array([[age, sex, cp, trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        pred = model.predict(arr)[0]
        return redirect(url_for('after', prediction=pred))
    return render_template('index.html')

@app.route("/after")
def after():
    prediction = request.args.get('prediction')
    return render_template('after.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
