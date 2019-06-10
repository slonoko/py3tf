#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

data = pd.read_csv('../dummy_data.csv')
data


# In[3]:


data.isnull().sum()


# In[5]:


data.values


# In[8]:


data.dropna(axis = 0)


# In[9]:


data.dropna(axis = 1)


# In[17]:


from sklearn.impute import SimpleImputer
import numpy as np
imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imp = imp.fit(data.values)
imputed_data = imp.transform(data.values)
imputed_data


# In[21]:


df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                          'ml/machine-learning-databases/'
                          'wine/wine.data', header=None)
from sklearn.model_selection import train_test_split
df_wine.columns = ['Class label', 'Alcohol',
                    'Malic acid', 'Ash',
                    'Alcalinity of ash', 'Magnesium',
                    'Total phenols', 'Flavanoids',
                    'Nonflavanoid phenols',
                    'Proanthocyanins',
                    'Color intensity', 'Hue',
                    'OD280/OD315 of diluted wines',
                    'Proline']
print('Class labels', np.unique(df_wine['Class label']))
print(df_wine.head())


# In[22]:


X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


# In[30]:

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

X_train_norm


# In[32]:


from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

X_train_std


# In[ ]:
from sklearn.preprocessing import MaxAbsScaler
abssc = MaxAbsScaler()
X_train_abs = abssc.fit_transform(X_train)
X_test_abs = abssc.transform(X_test)

X_train_abs





#%%
