#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import pickle
import socket

from flask import Flask,request


# In[2]:


os.chdir('C:/Users/sahil/Desktop/Learning/Python/Deployment/Docker/Bank Notes Classifier')


# In[3]:


app=Flask(__name__)
pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict')
def predict_note_authentication():
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[var,skewness,curtosis,entropy]])
    return "The predicted value is"+ str(prediction)

@app.route('/predict_file',methods=["POST"])
def predict_note_authentication():
    df_test=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df_test)
    return "The predicted value for the csv is"+ str(list(prediction))

if __name__=='__main__':
    app.run(debug=True)


# In[ ]:




