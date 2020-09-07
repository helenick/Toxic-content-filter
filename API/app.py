# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import flask
from flask import request
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_union
from timeit import default_timer as timer
import re, string
import joblib

#create flask framework main class
app = flask.Flask(__name__)
app.config["DEBUG"] = True

#define custom tokenizer
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \\1 ', s).split()

global vectorizer
global toxic_content_model
global obscene_content_model

#register POST request processor
@app.route('/check/comment', methods=['POST'])
def classify_comment():
    review = [request.form["review"]]#get review parameter
    test_features = vectorizer.transform(review)#trasform string to features
    result = "{Toxic: "
    val = toxic_content_model.predict(test_features)
    obs_val = obscene_content_model.predict(test_features)
    result = result + str(val[0])
    result = result + ", Obscene: " + str(obs_val[0]) + " }"
    return result#return result as JSON string

def init_models():
    global vectorizer
    global toxic_content_model
    global obscene_content_model

    #load models from files
    vectorizer = joblib.load("vectorizer.sav")
    toxic_content_model = joblib.load("finalized_model_toxic.sav")
    obscene_content_model = joblib.load("finalized_model_obscene.sav")

if __name__ == "__main__":
    init_models()
    app.run("127.0.0.1", "8080", debug=True)
