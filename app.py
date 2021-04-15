from flask import Flask, render_template, request
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from math import log, sqrt
import pandas as pd
import numpy as np
import re
from spam_ham import *
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html");   

@app.route('/predict', methods=['POST'])
def predict():
	mails = pd.read_csv('spam.csv', encoding = 'latin-1')
	mails.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
	mails.rename(columns = {'v1': 'labels', 'v2': 'message'}, inplace = True)
	mails['label'] = mails['labels'].map({'ham': 0, 'spam': 1})
	mails.drop(['labels'], axis = 1, inplace = True)

	totalMails = 4825 + 747
	trainIndex, testIndex = list(), list()
	for i in range(mails.shape[0]):
	    if np.random.uniform(0, 1) < 0.75:
	        trainIndex += [i]
	    else:
	        testIndex += [i]
	trainData = mails.loc[trainIndex]
	testData = mails.loc[testIndex]

	trainData.reset_index(inplace = True)
	trainData.drop(['index'], axis = 1, inplace = True)

	testData.reset_index(inplace = True)
	testData.drop(['index'], axis = 1, inplace = True)



    # df = pd.read_csv("spam.csv", encoding="latin-1")
    # df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    # # Features and Labels
    # df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    # X = df['message']
    # y = df['label']
    # # Extract Feature With CountVectorizer
    # cv = CountVectorizer()
    # X = cv.fit_transform(X)  # Fit the Data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # # Naive Bayes Classifier
    # clf = MultinomialNB()
    # clf.fit(X_train, y_train)
    # clf.score(X_test, y_test)

	sc_tf_idf = SpamClassifier(trainData, 'tf-idf')
	sc_tf_idf.train()

	if request.method == 'POST':
		message = request.form['message']
		pm = process_message(message)
		my_prediction = sc_tf_idf.classify(pm)
	return render_template('index.html', prediction=my_prediction)


if __name__ == '__main__':
   app.run(debug = True)
