import re
from flask import Flask, request, render_template, redirect, url_for
from app import app, APP_ROOT

from app.process import prognosis
# import os


@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        f = request.files['audio_data']
        with open('app/audio.wav', 'wb+') as audio:
            f.save(audio)

        return redirect(url_for('predict'))
    else:
        return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    # fname = request.form['First']
    # lname = request.form['last']
    # age = request.form['age']
   # country = request.form['inputCountry']
    # gender = request.form['inputGender']
   # print(age)

   # print(country)

    features = []
    for key in request.form:
        features.append(request.form[key])
    
    # print(len(features))

    x = prognosis('audio.wav', features)
    return render_template("output.html", result=x, len = len(features))


if __name__ == "__main__":
    app.run(debug=True)
