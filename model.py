# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 21:03:58 2019

@author: pathak
"""
from flask import Flask,render_template,url_for,request
import re  
import nltk  
nltk.download('stopwords') 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle

dataset = pd.read_csv('review.csv')
y = pd.get_dummies(dataset['stars'])
corpus = []  
for i in range(0, 10000):  
    review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])  
    review = review.lower()  
    review = review.split() 
    ps = PorterStemmer()    
    review = [ps.stem(word) for word in review 
                if not word in set(stopwords.words('english'))]
    review = ' '.join(review)  
    corpus.append(review)   
cv = CountVectorizer(max_features = 7000)  
X = cv.fit_transform(corpus).toarray() 
y = dataset.iloc[:, 1].values  
pickle.dump(cv, open('tranform.pkl', 'wb'))
from sklearn.cross_validation import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33) 
nb = MultinomialNB()
nb.fit(X_train, y_train)
filename = 'nlp_model.pkl'
pickle.dump(nb, open(filename, 'wb'))



