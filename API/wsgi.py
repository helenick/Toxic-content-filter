from app import app as webapp
from app import init_models
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_union
from timeit import default_timer as timer
import re, string
import joblib
import sys

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \\1 ', s).split()

current_module = sys.modules["__main__"]

current_module.tokenize = tokenize

init_models()

if __name__ == "__main__":
    webapp.run()