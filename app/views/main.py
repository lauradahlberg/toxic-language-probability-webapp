# import libs needed
from flask import render_template, jsonify, request, abort
from app import app

import pandas as pd, numpy as np
import re, string, random
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from scipy import sparse

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted

from joblib import dump, load

PATH = Path('./app')
MODELS = PATH/'views'

# load vectorizer
tfidf = load(MODELS/'tfidf_best_vect_2019-10-15.pkl')

# the models
lr_clf = load(MODELS/'logistic_regression_best_model_2019-10-15.pkl')

# to predict 
def predict_sent(sent, tfidf=tfidf, lr_clf=lr_clf):
    """ takes `sent` a sentence (str) and 
            models (`tfidf` is vectorizer, `lr_clf` is LogisticRegression model)
        returns the predicted probability of toxic statement as a number (rounded float)
    """
    # first preprocess sent
    import re
    # regex patterns
    re_punc = re.compile("([\"\''().,;:/_?!—\-“”¨«»®´·º½¾¿¡§£₤‘’])") # add spaces around punctuation " . "
    re_apos = re.compile(r"n ' t ")    # n't
    re_bpos = re.compile(r" ' s ")     # 's
    re_mult_space = re.compile(r"  *") # replace multiple spaces with just one " "
    re_nl = re.compile(r'\n') # ""
   
    # apply regex patterns
    sent = re_punc.sub(r" \1 ", sent)
    sent = re_apos.sub(r" n't ", sent)
    sent = re_bpos.sub(r" 's ", sent)
    sent = re_nl.sub(r" ", sent)
    sent = re_mult_space.sub(' ', sent)
   
    # lower-cased sent & strip whitespace off of ends
    # so TfidfVectorizer() can use its' default tokenization
    sent = sent.lower().strip()#.split()
   
    # return pred
    return round(lr_clf.predict_proba(tfidf.transform([sent]))[0][1],4)


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='Home')


@app.route('/result', methods=['POST'])
def result():
    # get user input sentence to analyze  
    query = request.form['query']
    sentence = query
    predictions = {}
    # predict toxic probability
    predictions['toxic probability'] = predict_sent(sent=sentence, tfidf=tfidf, lr_clf=lr_clf)
    return render_template('result.html', query=query, predictions=predictions, title='Result')

#commented out on layout.html b/c we don't need a map
@app.route('/map')
def map():
    return render_template('map.html', title='Map')


@app.route('/map/refresh', methods=['POST'])
def map_refresh():
    points = [(random.uniform(48.8434100, 48.8634100),
               random.uniform(2.3388000, 2.3588000))
              for _ in range(random.randint(2, 9))]
    return jsonify({'points': points})


@app.route('/contact')
def contact():
    return render_template('contact.html', title='Contact')