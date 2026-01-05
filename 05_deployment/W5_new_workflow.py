#!/usr/bin/env python
# coding: utf-8

# In the previous session we trained a model for predicting churn and evaluated it. Now let's deploy it

# In[2]:


get_ipython().run_line_magic('pip', 'install pandas numpy scikit-learn')


# In[3]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sklearn.pipeline import make_pipeline


# In[4]:


data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'


# In[5]:


get_ipython().system('wget $data -O data-week-3.csv')


# In[6]:


df = pd.read_csv('data-week-3.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# In[7]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# In[8]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]


# In[9]:


pipeline = make_pipeline(
    DictVectorizer(),
    LogisticRegression(C=1, max_iter=1000, solver = 'liblinear')
)


# In[10]:


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    pipeline.fit(dicts, y_train)

    return pipeline


# In[11]:


def predict(df, pipeline):
    dicts = df[categorical + numerical].to_dict(orient='records')
    y_pred = pipeline.predict_proba(dicts)[:, 1]

    return y_pred


# In[12]:


C = 1.0
n_splits = 5


# In[13]:


kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    pipeline  = train(df_train, y_train, C=C)
    y_pred = predict(df_val, pipeline )

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# In[14]:


scores


# In[15]:


pipeline  = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, pipeline )

y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)
auc


# Save the model using Pickel

# In[16]:


import pickle


# In[17]:


output_file = f'model_C = {C}.bin'
output_file


# In[18]:


#f_out = open(output_file, 'wb')
#pickle.dump((dv, model), f_out)
#f_out.close()

#better way of writing the above code is below, in that method, you  dont have to close manually


# In[19]:


with open(output_file, 'wb') as f_out:
  pickle.dump((pipeline), f_out)


# Load the model

# In[20]:


import pickle


# In[21]:


model_file = 'model_C = 1.0.bin'


# In[22]:


with open(model_file, 'rb') as f_in:
  pipeline = pickle.load(f_in)


# In[23]:


pipeline


# In[24]:


customer = {
    "gender": "male",
    "seniorcitizen": 0,
    "partner": "no",
    "dependents": "yes",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 6,
    "monthlycharges": 29.85,
    "totalcharges": 129.85
}


# In[25]:


pipeline.predict_proba(customer)[:, 1]

