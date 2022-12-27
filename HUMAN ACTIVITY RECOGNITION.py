#!/usr/bin/env python
# coding: utf-8

# ## HUMAN ACTIVITY RECOGNITION

# Name: ABISHEK N

# ### Step 1: Understand Data

# In[50]:


import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import precision_score, recall_score,accuracy_score,roc_auc_score,classification_report
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


# In[5]:


df = pd.read_csv(r"C:\Users\S.Saravanan\Desktop\dataset_pml\Human_Activity_Data.csv)


# In[6]:


#print top 5 rows
df.head()


# In[16]:


#print shape of the dataset
df.shape


# In[18]:


#print info() method
df.info()


# In[21]:


#print columns of the dataset
df.columns


# In[24]:


#type of the dataset
type(df)


# In[25]:


df['Activity'].value_counts


# ### **Step 2: Build a small dataset**

# In[29]:


lay = df.loc[df['Activity'] == "LAYING"][:500]
sit = df.loc[df['Activity'] == "SITTING"][:500]
walk = df.loc[df['Activity'] == "WALKING"][:500]

frames = [lay, sit, walk]

df_new = pd.concat(frames)


# In[30]:


df_new.shape


# In[31]:


df_new.to_csv("Human_Activity_sample.csv")


# In[32]:


df1= pd.read_csv('Human_Activity_sample.csv')


# In[33]:


df1.head()


# In[35]:


df1.shape


# In[37]:


df1.info()


# In[38]:


df1.columns


# In[39]:


type(df1)


# In[41]:


df1["Activity"].value_counts()


# ### Step 3: Build GradientBoostingClassifier

# In[42]:


X=df1.drop('Activity',axis=1)
y=df1.Activity


# In[43]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[44]:


model = GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1,random_state=0)
model.fit(X_train,y_train)


# In[46]:


y_pred=model.predict(X_test)


# In[47]:


accuracy_score(y_test,y_pred)


# In[51]:


print(classification_report(y_test,y_pred))


# ### Step4. [Find Best no. of trees and Best Learning Rate using Grid Search and Cross Validation]

# In[52]:


classifier = GradientBoostingClassifier()


# In[53]:


all_scores = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=5)


# In[54]:


all_scores


# ### To find the average of all the accuracies, simple use the mean() method

# In[55]:


all_scores.mean()


# In[56]:


parameter = {'n_estimators': [50, 100, 200, 400], 'learning_rate': [0.1, 0.01]}


# In[57]:


model1 = GridSearchCV(estimator=classifier, param_grid=parameter,cv=5, n_jobs=-1)


# In[58]:


model1.fit(X_train,y_train)


# In[81]:


y_pred2=model1.predict(X_test)


# In[82]:


accuracy_score(y_test,y_pred2)


# In[83]:


print(classification_report(y_test,y_pred2))


# In[62]:


print(model1.best_estimator_)


# ### Step5. [Build AdaBoostClassifier]

# In[63]:


base = DecisionTreeClassifier()


# In[64]:


model2 = AdaBoostClassifier(base_estimator=base,random_state=0)


# In[65]:


param_grid = {'n_estimators': [100, 150, 200], 'learning_rate': [0.01, 0.001]}


# In[66]:


model3 = GridSearchCV(model2,param_grid,cv=5,n_jobs=-1)


# In[67]:


model3.fit(X_train,y_train)


# In[68]:


y_pred3=model3.predict(X_test)


# In[69]:


accuracy_score(y_test,y_pred3)


# In[70]:


print(classification_report(y_test,y_pred3))


# In[71]:


print(model3.best_estimator_)


# ### Step6. [Build LogisticRegressionCV classifier]

# In[72]:


model4 = LogisticRegressionCV(cv=4,Cs=5,penalty='l2')


# In[73]:


model4.fit(X_train,y_train)


# In[77]:


y_pred4=model4.predict(X_test)


# In[84]:


accuracy_score(y_test,y_pred4)


# In[85]:


print(classification_report(y_test,y_pred4))


# ### Step 7 [ Build VotingClassifier]

# In[89]:


model5=VotingClassifier(estimators=[('lr',model4),('gbc',model1)], voting='hard')


# In[90]:


model5.fit(X_train,y_train)


# In[91]:


y_pred5=model5.predict(X_test)


# In[92]:


print(classification_report(y_test,y_pred5))


# ### Step8. [ Interpret your results]

# In[93]:


print(model1.best_estimator_)


# In[94]:


print(model3.best_estimator_)


# ## GradientBoostingClassifier

# ### GradientBoostingClassifier(n_estimators=50)

# In[95]:


classifierF = GradientBoostingClassifier(n_estimators=50)
all_scoresF = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=5)
parameter = {'n_estimators': [50, 100, 200, 400], 'learning_rate': [0.1, 0.01]}


# In[96]:


modelGC = GridSearchCV(estimator=classifier, param_grid=parameter,cv=5, n_jobs=-1)


# In[97]:


modelGC.fit(X_train,y_train)


# In[98]:


y_predGC=model3.predict(X_test)


# In[99]:


accuracy_score(y_test,y_predGC)


# In[100]:


print(classification_report(y_test,y_predGC))


# ## AdaBoostClassifier

# ### AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), learning_rate=0.01,n_estimators=100, random_state=0)

# In[101]:


modelABC = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), learning_rate=0.01,n_estimators=100, random_state=0)


# In[102]:


param_grid = {'n_estimators': [100, 150, 200], 'learning_rate': [0.01, 0.001]}


# In[103]:


modelGSCV = GridSearchCV(modelABC,param_grid,cv=5,n_jobs=-1)
modelGSCV.fit(X_train,y_train)


# In[104]:


y_predGSCV=model3.predict(X_test)


# In[105]:


accuracy_score(y_test,y_predGSCV)


# In[106]:


print(classification_report(y_test,y_predGSCV))

