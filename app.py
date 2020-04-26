# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 08:17:17 2020

@author: Suyog
"""

#Importing Libraries
import pandas as pd
import pickle
from flask import Flask,render_template,url_for,request

#Loading the model
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename,'rb')) #rb equals READ , BINARY
cv = pickle.load(open('transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
# =============================================================================
#         First you need to execute this part of code to generate 2 PICKLE file:  transform.pkl and nlp_model.pkl 
          #Importing dataset
#         data = pd.read_csv('spam.csv',encoding='latin-1')
#         data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
#         
#         #Features and Labels 
#         data['label'] = data['class'].map({'ham':0,'spam':1})
#         
#         #Splitting X and y
#         X = data['message']
#         y = data['label']
#         
#         #Extract Feature using CountVectorizer
#         from sklearn.feature_extraction.text import CountVectorizer
#         cv = CountVectorizer() #Serialization - convert object to byte stream
#         X = cv.fit_transform(X)
#         
#         pickle.dump(cv,open('transform.pkl','wb'))
#         
#         #Splitting train and test data set 
#         from sklearn.model_selection import train_test_split
#         X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
#         
#         #Applying Naive Bayes Classifier
#         from sklearn.naive_bayes import MultinomialNB
#         clf = MultinomialNB()
#         clf.fit(X_test,y_test)
#         filename = 'nlp_model.pkl'
#         pickle.dump(clf,open(filename,'wb'))
# 
# =============================================================================
    
    
    if request.method == 'POST':
        message = request.form['message']
        data =[message]
        vec = cv.transform(data).toarray()
        my_pred = clf.predict(vec)
    return render_template('result.html',prediction = my_pred)

if __name__ =='__main__':
    app.run(debug=True)