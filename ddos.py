#!/usr/bin/env python
# coding: utf-8

# 

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[23]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[25]:


df = pd.read_csv("./dataset_sdn.csv")
df.shape


# In[26]:


df.sample(5)


# In[27]:


df.info()


# In[28]:


df.isnull().sum()


# In[29]:


df = df.dropna()


# In[30]:


df.describe()


# In[31]:


numeric_df = df.select_dtypes(include=['int64', 'float64'])
object_df = df.select_dtypes(include=['object'])
numeric_cols = numeric_df.columns
object_cols = object_df.columns
print('Numeric Columns: ')
print(numeric_cols, '\n')
print('Object Columns: ')
print(object_cols, '\n')
print('Number of Numeric Features: ', len(numeric_cols))
print('Number of Object Features: ', len(object_cols))


# In[32]:


label_encoder = LabelEncoder()
df['src'] = label_encoder.fit_transform(df['src'])
df['dst'] = label_encoder.fit_transform(df['dst'])
df['Protocol'] = label_encoder.fit_transform(df['Protocol'])


# In[33]:


X = df.drop('label', axis=1)
y = df['label']


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[35]:


def Evalute_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy Score: {:.2f}%".format(accuracy_score(y_test, y_pred)*100))
    print("Classification Report:\n{}".format(classification_report(y_test, y_pred))) 
    print("Confusion Matrix:\n{}".format(confusion_matrix(y_test, y_pred)))


# In[36]:


Evalute_model(LogisticRegression())


# In[37]:


Evalute_model(RandomForestClassifier())


# In[38]:


Evalute_model(DecisionTreeClassifier())


# In[39]:


Evalute_model(GaussianNB())

