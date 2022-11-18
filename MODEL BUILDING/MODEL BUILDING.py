#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
import numpy as np
import matplotlib as mlt
import sklearn
import scipy
import seaborn as sb
import missingno as msno


# In[35]:


data=pd.read_csv("C:\\Users\\STUDENT\\Downloads\\weatherAUS.csv")


# In[36]:


data.describe()


# In[37]:


data.info()


# In[38]:


data.isnull().sum()


# In[ ]:


import missingno as msno


# In[39]:


msno.matrix(data,color=(0.55,0.255,0.255),fontsize=16)


# In[40]:


data_c=data[["RainToday","WindGustDir","WindDir9am","WindDir3pm"]]


# In[41]:



data.drop(columns=["Evaporation","Sunshine","Cloud9am","Cloud3pm"],axis=1,inplace=True)
data.drop(columns=["RainToday","WindGustDir","WindDir9am","WindDir3pm"],axis=1,inplace=True)


# In[42]:


data['MinTemp'].fillna(data['MinTemp'].mean(),inplace=True)
data['MaxTemp'].fillna(data['MaxTemp'].mean(),inplace=True)
data['Rainfall'].fillna(data['Rainfall'].mean(),inplace=True)
data['WindGustSpeed'].fillna(data['WindGustSpeed'].mean(),inplace=True)
data['WindSpeed9am'].fillna(data['WindSpeed9am'].mean(),inplace=True)
data['WindSpeed3pm'].fillna(data['WindSpeed3pm'].mean(),inplace=True)
data['Humidity3pm'].fillna(data['Humidity3pm'].mean(),inplace=True)
data['Humidity9am'].fillna(data['Humidity9am'].mean(),inplace=True)
data['Temp9am'].fillna(data['Temp9am'].mean(),inplace=True)
data['Temp3pm'].fillna(data['Temp3pm'].mean(),inplace=True)


# In[43]:


c_names=data_c.columns


# In[44]:


from sklearn.impute import SimpleImputer


# In[45]:



from sklearn.impute import SimpleImputer
imp_mode=SimpleImputer(missing_values=np.nan,strategy="most_frequent")


# In[46]:



data_c=imp_mode.fit_transform(data_c)


# In[47]:


data_c=pd.DataFrame(data_c,columns=c_names)


# In[48]:


data_c.tail()


# In[49]:



data.head()


# In[50]:


data=pd.concat([data,data_c],axis=1)


# In[51]:


data.head()


# In[52]:


corr=data.corr()


# In[53]:


sb.heatmap(data=corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)


# In[54]:


sb.jointplot(data["MinTemp"],data['Rainfall'])


# In[55]:


data.boxplot()


# In[60]:


sb.scatterplot(data['MaxTemp'],data['Rainfall'])


# In[61]:


sb.displot(data['MinTemp'])


# In[62]:



from sklearn.preprocessing import StandardScaler


# In[63]:


data = data[data['RainTomorrow'].notnull()]


# In[64]:


data['Pressure9am'].fillna(data['Pressure9am'].mean(),inplace=True)
data['Pressure3pm'].fillna(data['Pressure3pm'].mean(),inplace=True)


# In[65]:



y=data['RainTomorrow']
x=data.drop('RainTomorrow',axis=1)


# In[66]:


set(y)


# In[67]:


x=x.drop('Date',axis=1)


# In[68]:


names=x.columns


# In[69]:


names


# In[70]:


sc=StandardScaler()


# In[71]:


from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# In[72]:


print(len(x),len(y))


# In[73]:


## RainToday	WindGustDir	WindDir9am	WindDir3pm

LE = LabelEncoder()
x['Location'] = LE.fit_transform(x['Location'])
x.head()

LE = LabelEncoder()
x['RainToday'] = LE.fit_transform(x['RainToday'])
x.head()

LE = LabelEncoder()
x['WindGustDir'] = LE.fit_transform(x['WindGustDir'])
x.head()

LE = LabelEncoder()
x['WindDir9am'] = LE.fit_transform(x['WindDir9am'])
x.head()

LE = LabelEncoder()
x['WindDir3pm'] = LE.fit_transform(x['WindDir3pm'])
x.head()


# In[74]:


LE = LabelEncoder()
y=pd.DataFrame(y)
y = LE.fit_transform(y)


# In[75]:



print(len(x),len(y))


# In[76]:


sc=StandardScaler()


# In[77]:


x=sc.fit_transform(x)


# In[78]:


x[:5]


# In[79]:



x=pd.DataFrame(x,columns=names)


# In[81]:


from sklearn import model_selection


# In[82]:


x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2,random_state=0)


# In[84]:


from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier


# In[85]:


RFC=RandomForestClassifier()


# In[86]:


GBC=GradientBoostingClassifier()


# In[87]:



np.any(np.isnan(x))


# In[88]:


GBC.fit(x_train,y_train)


# In[89]:


GradientBoostingClassifier()


# In[90]:


RFC.fit(x_train,y_train)


# In[91]:


RandomForestClassifier()


# In[92]:


data.isnull().any()


# In[93]:


x.isnull().any()


# In[94]:


p1=RFC.predict(x_train)


# In[95]:


p2=RFC.predict(x_test)


# In[96]:


import sklearn.metrics as metrics


# In[97]:


print(metrics.accuracy_score(y_train,p1))


# In[98]:


print(metrics.accuracy_score(y_test,p2))


# In[99]:


import pickle


# In[100]:


pickle.dump(RFC,open('rainfall.pkl','wb'))
pickle.dump(LE,open('encoder.pkl','wb'))
pickle.dump(imp_mode,open('imputer.pkl','wb'))
pickle.dump(sc,open('scale.pkl','wb'))


# In[ ]:




