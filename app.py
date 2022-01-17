from flask import Flask, request, render_template
from flask_cors import CORS
import flask
import os

from Prediction import predict_news

#Loading Flask and assigning the model variable
app = Flask(__name__)
CORS(app)
app = flask.Flask(__name__, template_folder = 'templates')

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    url = request.get_data(as_text = True)[8:]
    prediction = predict_news(url)
    truth_value = prediction[0]
    truth_probability = prediction[1]
    final_probability = truth_probability if str(truth_value) == 'True' else (1 - truth_probability)
    result = 'The statement is ' + str(truth_value) + ' with probability ' + str(final_probability)
    return render_template('index.html', prediction_text = result)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port = port, debug = True, use_reloader = False)
