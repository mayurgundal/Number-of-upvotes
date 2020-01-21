#!/usr/bin/env python
# coding: utf-8

# In[17]:


import sys
import numpy as np
import pandas as pd
import sklearn 
import matplotlib.pyplot as plt
import scipy as sci
import seaborn as sns


# In[2]:


data=pd.read_csv(r'F:\machine learning\analytics_vidya\predict_upvotes\train_NIR5Yl1 (1).csv')


# In[3]:


#data=pd.read_csv(r'F:\M.tech_data\sea surface temperatue.csv')


# In[4]:


print(data.columns)


# In[5]:


data.head(10)


# In[ ]:





# In[6]:


data.shape


# In[7]:


print(data.describe())


# In[8]:


data.isnull().sum()


# In[9]:


data.tail()


# In[10]:


corrmat=data.corr()
fig=plt.figure(figsize=(12,9))

seaborn.heatmap(corrmat,vmax=0.8,square=True)
plt.show()


# In[11]:


data['Tag'].value_counts().head(10).plot.bar()


# In[12]:


data['Upvotes'].value_counts().head(10).plot.bar()


# In[13]:


data['Reputation'].value_counts().head(10).plot.bar()


# In[ ]:


sns.barplot(x=data_f["tag1"], y=data["Upvotes"])


# In[19]:


data['tag1'] = data['Tag'].map( {'c':7, 'j':10, 'p':9, 'i':4, 'a':6, 's':5, 'h':8, 'o':2, 'r':1, 'x':3} )
data[['Tag', 'tag1']]


# In[20]:


data.head()


# In[21]:


corrmat=data.corr()
fig=plt.figure(figsize=(12,9))

seaborn.heatmap(corrmat,vmax=0.8,square=True)
plt.show()


# In[22]:


data_f=data[['Reputation','Answers','Username','Views','tag1']]


# In[23]:


data_f.head()


# In[24]:


import matplotlib.pyplot as plt


# In[25]:


plt.scatter(data_f['tag1'],data['Upvotes'] )
plt.show()


# In[26]:


import seaborn as sns


# In[27]:


sns.barplot(x=data_f["tag1"], y=data["Upvotes"])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[28]:


data_f['ans_view']=data_f['Answers']*data_f['Views']


# In[29]:


data_f['perce_answer_reputa']= (data_f['Reputation']-data_f['Answers'])/100


# In[30]:


data_f['ans_tag']=data_f['Answers']*data_f['tag1']


# In[31]:


data_f['Reptn_view']=data_f['Reputation']*data_f['Views']/10000


# In[32]:


data_f['Reput_username']=data_f['Reputation']*data_f['Username']/10000


# In[33]:


data_f['tag_view']=data_f['tag1']*data_f['Views']/100


# In[34]:


data_f['tag1_usename']=data_f['tag1']*data_f['Username']/100


# In[35]:


data_f['tag_rupt']=data_f['tag1']*data_f['Reputation']/100


# In[36]:


data_f['view_username']=data_f['Reputation']*data_f['Views']/10000


# In[37]:


data_f['view_view']=data_f['Views']*data_f['Views']/100000000


# In[38]:


data_f['ans_ans']=data_f['Answers']*data_f['Answers']/100


# In[39]:


#data_f['ans_view']=data_f['Answers']*data_f['Views']/10000


# In[40]:


data_f.head()


# In[41]:


Y=data[['Upvotes']]


# In[42]:


Y.head()


# In[43]:


import sklearn
import pandas as pd
#import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle


# In[44]:


X=np.array(data_f)
Y=np.array(Y)


# In[45]:


X=preprocessing.scale(X)


# In[46]:


Y=preprocessing.scale(Y)


# In[47]:


X_train,X_test,Y_train,Y_test= train_test_split( X, Y, test_size=0.3)


# In[48]:


clf=LinearRegression()


# In[49]:


clf.fit(X_train, Y_train)


# In[50]:


#coeff_df = pd.DataFrame(clf.coef_, X.columns, columns=['Coefficient'])  
#coeff_df


# In[51]:


#y_pred = clf.predict(X_test)


# In[52]:


#Y_train.shape
#y_pred.shape
#Y_test.shape


# In[53]:


#clf.score(Y_test,y_pred)


# In[54]:


#import matplotlib.pyplot as plot

#plot.scatter(X_train, Y_train, color = 'red')
#plot.plot(X_train, clf.predict(X_train), color = 'blue')
#plot.title('Salary vs Experience (Training set)')
#plot.xlabel('Years of Experience')
#plot.ylabel('Salary')
#plot.show()


# In[ ]:





# In[55]:


#df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})


# In[56]:


#with open('linearRegression.pickle', 'wb') as f:
#  pickle.dump(clf, f)


# In[57]:


#pickle_in= open('linearRegression.pickle', 'rb')
#clf=pickle.load(pickle_in)


# In[58]:


accuracy=clf.score(X_test, Y_test)


# In[59]:


print(accuracy)


# In[60]:


accuracy1=clf.score(X_train, Y_train)


# In[61]:


print(accuracy1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[62]:


from sklearn.ensemble import GradientBoostingRegressor


# In[63]:


gb = GradientBoostingRegressor(random_state=0,max_depth=8, alpha=0.2, learning_rate=0.0999, min_samples_leaf=12,
                               max_features=15,n_estimators=50)


# In[64]:


gb.fit(X_train,Y_train)


# In[65]:


accuracy1=gb.score(X_train, Y_train)


# In[66]:


print(accuracy1)


# In[67]:


accuracy=gb.score(X_test, Y_test)


# In[68]:


print(accuracy)


# In[ ]:





# In[ ]:





# In[ ]:





# In[69]:


from sklearn.svm import SVR


# In[70]:


svr=SVR()


# In[ ]:


svr.fit(X_train,Y_train)


# In[ ]:


accr=svr.score(X_train,Y_train)


# In[ ]:


print(accr)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rf = RandomForestRegressor(max_features=12, min_samples_split=10, n_estimators=50, min_samples_leaf=6)


# In[ ]:


rf.fit(X_train,Y_train)


# In[77]:


accuracy_rf=rf.score(X_train,Y_train)


# In[78]:


print(accuracy_rf)


# In[79]:


accuracy_rf_test=rf.score(X_test,Y_test)


# In[80]:


print(accuracy_rf_test)


# In[ ]:





# In[ ]:





# In[49]:


data_1=pd.read_csv(r'F:\machine learning\analytics_vidya\predict_upvotes\test_8i3B3FC.csv')


# In[50]:


data_1.shape


# In[51]:


data_1.head()


# In[52]:


data_1['tag1'] = data_1['Tag'].map( {'c':1, 'j':2, 'p':3, 'i':4, 'a':5, 's':6, 'h':7, 'o':8, 'r':9, 'x':10} )
data_1[['Tag', 'tag1']]


# In[53]:


data_f1=data_1[['Reputation','Answers','Username','Views','tag1']]


# In[54]:


print(np.unique(data_1['tag1']))


# In[ ]:





# In[55]:


data_f1['ans_view']=data_f1['Answers']*data_f1['Views']


# In[56]:


data_f1['perce_answer_reputa']= (data_f1['Reputation']-data_f1['Answers'])/100


# In[57]:


data_f1['ans_tag']=data_f1['Answers']*data_f1['tag1']


# In[58]:


data_f1['Reptn_view']=data_f1['Reputation']*data_f1['Views']/10000


# In[59]:


data_f1['Reput_username']=data_f1['Reputation']*data_f1['Username']/10000


# In[60]:


data_f1['tag_view']=data_f1['tag1']*data_f1['Views']/100


# In[61]:


data_f1['tag1_usename']=data_f1['tag1']*data_f1['Username']/100


# In[62]:


data_f1['tag_rupt']=data_f1['tag1']*data_f1['Reputation']/100


# In[63]:


data_f1['view_username']=data_f1['Reputation']*data_f1['Views']/10000


# In[64]:


data_f1.head()


# In[65]:


X_test_1=np.array(data_f1)


# In[178]:


#X_test_1=preprocessing.scale(X_test_1)


# In[77]:


aa=rf.predict(X_test_1)


# In[78]:


aa.shape


# In[86]:


jj=[]


# In[87]:


for i in aa:
    jj.append(int(round(i)))
    


# In[88]:


jj[1:10]


# In[89]:


final_sbmt= pd.DataFrame({'ID': data_1['ID'], 'Upvotes':jj})


# In[90]:


final_sbmt


# In[ ]:





# In[91]:


#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = ' Predict number of upvotes44.csv'

final_sbmt.to_csv(filename,index=False)

print('Saved file: ' + filename)


# In[ ]:





# In[ ]:





# In[63]:


from sklearn.tree import DecisionTreeRegressor


# In[64]:


ada_tree_backing = DecisionTreeRegressor()


# In[65]:


ada_tree_backing.fit(X_train,Y_train)


# In[66]:


accuracy_ada=ada_tree_backing.score(X_train,Y_train)


# In[67]:


print(accuracy_ada)


# In[68]:


accuracy_ada_test=ada_tree_backing.score(X_test,Y_test)


# In[ ]:


print(accuracy_ada_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


clf1=svm.SVR()


# In[ ]:


clf1.fit(X_train,Y_train)
accuracy12=clf1.score(X_test,Y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[203]:


data_12=pd.read_csv(r'F:\machine learning\analytics_vidya\predict_upvotes\sample_submission_OR5kZa5.csv')


# In[205]:


data_12


# In[ ]:




