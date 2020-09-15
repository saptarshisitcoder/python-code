# python-code
code keep


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score,mean_squared_error

data=pd.read_csv('GDP_Country.csv')
data.head()

data.set_index(['Country','Region'],inplace=True)
data.head()

data.isnull().sum()

from sklearn.impute import KNNImputer

fill_mod=KNNImputer(n_neighbors=3)
data_fill=fill_mod.fit_transform(data)
data_fill=pd.DataFrame(data_fill)
data_fill.columns=data.columns
data_fill.index=data.index

out=data_fill['GDP ($ per capita)']
inp=data_fill.drop('GDP ($ per capita)',axis=1)
inp_c=sm.add_constant(inp)
result=sm.OLS(out,inp_c).fit()
result.summary()

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

vf=[vif(inp.values,i) for i in range(inp.shape[1])]
pd.DataFrame(vf,index=inp.columns,columns=['vif'])

