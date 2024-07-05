#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# to visualize the dataset
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# To preprocess the data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
# import iterative imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
# machine learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# for classification tasks
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
#metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error, precision_score, r2_score, mean_squared_error
# impot pipeline
from sklearn.pipeline import Pipeline
# ignore warnings   
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


# In[2]:


df = pd.read_csv(r"C:\Users\shashank\Downloads\heart_disease_uci.csv")


# In[3]:


#Checking the dataset:


# In[4]:


print(df.shape)


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


print(df.isnull().sum().sort_values(ascending=False))


# In[8]:


#Performing EDA


# In[9]:


sns.histplot(df['age'], kde=True)
plt.axvline(df['age'].mean(), color='red')
plt.axvline(df['age'].median(), color='green')
plt.axvline(df['age'].mode()[0], color='blue')

# print the value of mean, median and mode of age column
print('Mean:', df['age'].mean())
print('Median:', df['age'].median())
print('Mode:', df['age'].mode()[0])


# In[10]:


fig = px.histogram(data_frame=df, x='age', color='sex')
fig.show()


# In[11]:


print(df['sex'].value_counts())


# In[12]:


male_count = 726
female_count = 194
total_count = male_count + female_count

# calculate percentages
male_percentage = (male_count / total_count) * 100
female_percentage = (female_count / total_count) * 100

# display the results
print(f"Male percentage in the data: {male_percentage:.2f}%")
print(f"Female Percentage in the data: {female_percentage:.2f}%")


# In[13]:


df['sex'].value_counts().plot(kind='pie',autopct='%2.f')


# In[14]:


df['dataset'].unique()


# In[15]:


df['dataset'].value_counts()


# In[16]:


print(df.groupby('sex')['dataset'].value_counts())


# In[17]:


df.groupby('dataset')['num'].value_counts().plot(kind='bar')


# In[18]:


df['trestbps'].describe()


# In[19]:


print(sns.histplot(df['trestbps'], kde=True))


# In[20]:


# make a plot of trestbps column using plotly and coloring this by dataset column
fig = px.histogram(data_frame=df, x='trestbps', color='dataset')
fig.show()


# In[21]:


df.groupby(df['sex'])[['trestbps']].describe()


# In[22]:


sns.histplot(df['chol'], kde=True)
plt.axvline(df['chol'].mean(), color='red')
plt.axvline(df['chol'].median(), color='green')
plt.axvline(df['chol'].mode()[0], color='blue')


# In[23]:


df.groupby(df['sex'])[['chol']].describe()


# In[24]:


df['fbs'].info()


# In[25]:


df['fbs'].value_counts()


# In[26]:


df.groupby(df['fbs'])[['sex']].value_counts()


# In[27]:


df['restecg'].info()


# In[28]:


df['restecg'].value_counts()


# In[29]:


df.groupby(df['restecg'])[['sex']].value_counts()


# In[30]:


df.groupby(df['restecg'])[['dataset']].value_counts()


# In[31]:


counts = df.groupby(df['restecg'])[['dataset','sex']].value_counts().unstack()
print(counts.plot.bar())


# In[32]:


df['thalch'].info()


# In[33]:


print(sns.histplot(df['thalch'],kde = True))


# In[34]:


# `chol` (serum cholesterol in mg/dl)
sns.histplot(df['thalch'], kde=True)
plt.axvline(df['thalch'].mean(), color='red')
plt.axvline(df['thalch'].median(), color='green')
plt.axvline(df['thalch'].mode()[0], color='blue')


# In[35]:


df['exang'].value_counts()


# In[36]:


df.groupby(df['exang'])[['sex']].value_counts()


# In[37]:


df['oldpeak'].describe()


# In[38]:


df['slope'].value_counts()


# In[39]:


df.groupby(df['slope'])['restecg'].value_counts()


# In[40]:


df['ca'].value_counts()


# In[41]:


print(df['num'].unique())


# In[42]:


df['num'].value_counts()


# In[43]:


df.groupby(df['num'])[['dataset']].value_counts()


# In[44]:


#Understading the correlation


# In[45]:


sns.heatmap(df.corr(),annot=True)


# In[46]:


from ydata_profiling import ProfileReport


# In[47]:


prof = ProfileReport(df)


# In[48]:


prof.to_file(output_file='mainfinal.html')


# In[49]:


df['restecg'] = df['restecg'].apply(lambda x:0 if x=='normal' else( 1 if x=='lv hypertrophy' else 2) )


# In[50]:


df['slope']= df['slope'].apply(lambda x:0 if x=='flat' else( 1 if x=='upsloping' else 2) )


# In[51]:


df.drop(['id','dataset'],axis=1,inplace=True)


# In[52]:


#handling The Outliers
df


# In[53]:


import seaborn as sns
import pandas as pd

# Example data
data = pd.DataFrame({
    "Age": df['age'],
    "trestbps": df['trestbps'],
    "chol": df['chol'],
    "thalch":df['thalch'],
    "oldpeak":df['oldpeak'],
    "slope":df['slope'],
    "ca":df['ca']
    
})

# Plotting boxplots
sns.boxplot(data=data)


# In[54]:


plt.figure(figsize=(16,8))
plt.subplot(2,3,1)
sns.boxplot(df['age'])

plt.subplot(2,3,2)
sns.boxplot(df['trestbps'])

plt.subplot(2,3,3)
sns.boxplot(df['chol'])

plt.subplot(2,3,4)
sns.boxplot(df['thalch'])

plt.subplot(2,3,5)
sns.boxplot(df['oldpeak'])

plt.subplot(2,3,6)
sns.boxplot(df['ca'])

# Show the plots
plt.show()


# In[55]:


plt.figure(figsize=(16,8))
plt.subplot(2,3,1)
sns.distplot(df['age'])

plt.subplot(2,3,2)
sns.distplot(df['trestbps'])

plt.subplot(2,3,3)
sns.distplot(df['chol'])

plt.subplot(2,3,4)
sns.distplot(df['thalch'])

plt.subplot(2,3,5)
sns.distplot(df['oldpeak'])

plt.subplot(2,3,6)
sns.distplot(df['ca'])

# Show the plots
plt.show()


# In[56]:


upperthalch = df['thalch'].mean()+3*df['thalch'].std()
upperthalch


# In[57]:


lowerthalch = df['thalch'].mean() - 3 * df['thalch'].std()
lowerthalch


# In[58]:


df[(df['thalch']>215.32) | (df['thalch']<5)]


# In[59]:


df['age'].skew()


# In[60]:


df['trestbps'].skew()


# In[61]:


df['chol'].skew()


# In[62]:


df['thalch'].skew()


# In[63]:


df['oldpeak'].skew()


# In[64]:


df['ca'].skew()


# In[65]:


p25t = df['trestbps'].quantile(0.25)
p75t = df['trestbps'].quantile(0.75)
iqrt = p75t-p25t
uppert = p75t+iqrt
lowert = p25t-iqrt


# In[66]:


uppert


# In[67]:


lowert


# In[68]:


df['trestbps'] = np.where(df['trestbps']>uppert,uppert,np.where(
df['trestbps']<lowert,lowert,df['trestbps']))


# In[69]:


p25c = df['chol'].quantile(0.25)
p75c = df['chol'].quantile(0.75)
iqrc = p75c-p25c
upperc = p75c+iqrc
lowerc = p25c-iqrc


# In[70]:


upperc


# In[71]:


lowerc


# In[72]:


df['chol'] = np.where(df['chol']>upperc,upperc,np.where(
df['chol']<lowerc,lowerc,df['chol']))


# In[73]:


p25o = df['oldpeak'].quantile(0.25)
p75o = df['oldpeak'].quantile(0.75)
iqro = p75o-p25o
uppero = p75o+iqro
lowero = p25o-iqro


# In[74]:


uppero


# In[75]:


lowero


# In[76]:


df['oldpeak'] = np.where(df['oldpeak']>uppero,uppero,np.where(df['oldpeak']<lowero,lowero,df['oldpeak']))


# In[77]:


plt.figure(figsize=(16,8))
plt.subplot(2,3,1)
sns.boxplot(df['age'])

plt.subplot(2,3,2)
sns.boxplot(df['trestbps'])

plt.subplot(2,3,3)
sns.boxplot(df['chol'])

plt.subplot(2,3,4)
sns.boxplot(df['thalch'])

plt.subplot(2,3,5)
sns.boxplot(df['oldpeak'])

plt.subplot(2,3,6)
sns.boxplot(df['ca'])

# Show the plots
plt.show()


# In[78]:


df


# In[79]:


X_train,X_test,Y_train,Y_test = train_test_split(df.drop(['num'],axis=1),df['num'])


# In[176]:


trf2= ColumnTransformer(transformers = [
    ('impute_bps_Chol_thal_ch_oldPeak_ca',SimpleImputer(),[3,4,7,9,11]),
    ('impute_fbs_restecg_exang_slope',SimpleImputer(strategy='most_frequent'),[5,6,8,10]),
    ('oc',OrdinalEncoder(categories=[['asymptomatic','non-anginal','atypical angina','typical angina']]),[2]),
    ('oh_sex_fbs_exang',OneHotEncoder(sparse=False,handle_unknown='ignore'),[1,5,8]),
],remainder='passthrough')


# In[177]:


trf3 = StandardScaler()


# In[178]:


pipe = make_pipeline(trf2,trf3)


# In[179]:


from sklearn.ensemble import GradientBoostingClassifier


# In[180]:


from sklearn.model_selection import GridSearchCV


# In[102]:


from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,100,1000]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    },
    'knearestneighbors' : {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [1,3,5,30]
        }
    },
    'naive_bayes':{
        'model':GaussianNB(),
        'params':{
            
        }
    }
}


# In[103]:


scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=2, return_train_score=False)
    X = pipe.fit_transform(X_train)
    clf.fit(X,Y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
scores = pd.DataFrame(scores,columns=['model','best_score','best_params'])
scores


# In[95]:


'XGBClassifier': {
        'model':GradientBoostingClassifier(), 
        'params':{
            'n_estimators': [10, 100, 1000], 
            'learning_rate': [0.1, 0.01, 0.001]
        }
    }


# In[181]:


X = df.drop('num',axis=1)
y = df['num']
# replace 1 with 0 and all others to 1
y = np.where((y == 1) | (y == 2) | (y == 3) | (y == 4), 1,0)


# In[182]:


X_train,X_test,Y_train,Y_test = train_test_split(X,y)


# In[183]:


from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

model_params = {
    'svm': {
        'model': svm.SVC(),
        'params' : {
    
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,2,3]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(penalty='l2'),
        'params': {
            
        }
    },
    'knearestneighbors' : {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [1,3,5,30]
        }
    },
    'naive_bayes':{
        'model':GaussianNB(),
        'params':{
            
        }
    }
}


# In[184]:


scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=2, return_train_score=False)
    X = pipe.fit_transform(X_train)
    clf.fit(X,Y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
scores = pd.DataFrame(scores,columns=['model','best_score','best_params'])
scores


# In[185]:


import joblib


# In[186]:


trf5 = LogisticRegression()


# In[187]:


last_pipe = make_pipeline(trf2,trf3,trf5)


# In[189]:


last_pipe.fit(X_train,Y_train)


# In[190]:


last_pipe.predict([[63,'Male','typical angina',145.0,233.0,True,1,150.0,False,2.3,2,0.0]])


# In[191]:


joblib.dump(last_pipe,'major_finalModel.pkl')


# In[ ]:




