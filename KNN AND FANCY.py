#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer


# In[4]:


df = pd.read_csv("synthetic_dataset.csv")
df


# In[5]:


df.isna()


# In[6]:


df.isna().sum()


# In[7]:


df_imputer_knn_sklearn = df.copy(deep = True)
df_imputer_knn_sklearn


# In[8]:


df_imputer_knn_sklearn.isna().sum()


# In[22]:


num_cols = ['age', 'purchase_amount_kes', 'quantity', 'satisfaction_score']
df_num = df[num_cols].copy()
df_num


# In[25]:


knn_imp = KNNImputer(n_neighbors = 5)
df_knn = pd.DataFrame(knn_imp.fit_transform(df_num), columns = num_cols)
df_knn


# In[29]:


from sklearn.impute import SimpleImputer
cat_cols = df.select_dtypes(include ='object').columns
cat_imp = SimpleImputer(strategy = 'most_frequent')
df_cat = pd.DataFrame(cat_imp.fit_transform(df[cat_cols]), columns = cat_cols)
df_cat


# In[30]:


df_final = pd.concat([df_knn,df_cat], axis = 1)
df_final


# In[35]:


df_final = df_final[df.columns]
df_final


# In[ ]:




