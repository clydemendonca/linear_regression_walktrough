
# coding: utf-8

# In[81]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


# In[2]:


df = pd.read_csv('./train.csv')


# In[3]:


df.head()


# In[5]:


df.columns


# In[6]:


len(df.columns)


# ## 80-ish features is too much for Linear Regression

# In[8]:


df.info()


# ## Get numeric columns correlation matrix

# In[14]:


corr = df.select_dtypes(include=[np.number]).corr()


# In[16]:


corr


# In[24]:


corr[['SalePrice']]


# In[28]:


corr.loc[(corr['SalePrice'].abs() >= 0.6), ['SalePrice']]


# In[114]:


high_correlation_with_target_columns =  corr.loc[(corr['SalePrice'].abs() >= 0.6), ['SalePrice']].index
high_correlation_with_target_columns


# ## Remove sale price from columns

# In[115]:


high_correlation_with_target_columns = high_correlation_with_target_columns[:-1]
high_correlation_with_target_columns


# ## Of the columns which are hightly correlated with the target, find the correlation with each other

# In[116]:


refined_corr = corr.loc[ high_correlation_with_target_columns,:]
refined_corr = refined_corr[high_correlation_with_target_columns]


# In[117]:


refined_corr


# In[118]:


refined_corr


# In[119]:


refined_corr < .5


# In[120]:


refined_corr_boolean = refined_corr < .5
refined_corr_boolean = refined_corr_boolean.replace(False, '')
refined_corr_boolean


# ## We choose OverallQual, TotalBsmtSF, GrLivArea, GarageArea

# In[121]:


def linear_regression(columns_list):
    X = df[columns_list]
    y = df[['SalePrice']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    score = regressor.score(X_test, y_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return score, rmse


# In[122]:


columns_list = ['OverallQual', 'TotalBsmtSF', 'GrLivArea', 'GarageArea']

linear_regression(columns_list)


# In[123]:


columns_list = ['OverallQual', 'TotalBsmtSF', 'GrLivArea', 'GarageCars', 'GarageArea']

linear_regression(columns_list)


# In[124]:


columns_list = ['OverallQual', 'TotalBsmtSF', 'GrLivArea', 'GarageCars', 'GarageArea', '1stFlrSF']

linear_regression(columns_list)

