
# coding: utf-8

# In[1]:


#!/usr/bin/env python

import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[2]:


# Reading data from csv file
data = pd.read_csv("~/datascience/data.csv")
data.head()


# In[3]:


# Labels
y = data["label"]

# Features
url_list = data["url"]


# In[4]:


# Using Tokenizer
vectorizer = TfidfVectorizer()

# Store vectors into X variable as Our XFeatures
X = vectorizer.fit_transform(url_list)


# In[5]:


# Split into training and testing dataset 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


# Model Building using logistic regression
logit = LogisticRegression()
logit.fit(X_train, y_train)


# In[7]:


# Accuracy of Our Model
print("Accuracy of our model is: ",logit.score(X_test, y_test))

