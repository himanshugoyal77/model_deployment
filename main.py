from flask import Flask,request, url_for, redirect, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model_x = pickle.load(open('new_modal.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route("/predict", methods =["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model_x.predict(features)

    return render_template('index.html', prediction_text = "The % prediction is {}".format(prediction))


if __name__ == '__main__':
    app.run(debug=True)