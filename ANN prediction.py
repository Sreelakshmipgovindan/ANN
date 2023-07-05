#!/usr/bin/env python
# coding: utf-8

# # import the libraries

# In[1]:


#import pandas
import pandas as pd
#import numpy
import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
#import seaborn
import seaborn as sb


# # loading the dataset

# In[5]:


# use pandas to import csv file
df = pd.read_csv ('WA_Fn-UseC_-Telco-Customer-Churn.csv') 


# In[6]:


pd.set_option('display.max_columns',None)
# print dataframe
df


# # data processing

# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df = df.drop('customerID',axis=1)


# In[10]:


#count of string value into the column.
count=0
for i in df.TotalCharges:
    if i==' ':
        count+=1
print('count of empty string:- ',count)
#we will replace this empty string to nan values
df['TotalCharges'] = df['TotalCharges'].replace(" ",np.nan)
# typecasting of the TotalCharges column
df['TotalCharges'] = df['TotalCharges'].astype(float)


# In[11]:


df.isnull().sum()


# In[12]:


df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())


# In[13]:


#numerical variables

num = list(df.select_dtypes(include=['int64','float64']).keys())

#categorical variables

cat = list(df.select_dtypes(include='O').keys())

print(cat)

print(num)


# In[14]:


# value_counts of the categorical columns
for i in cat:
    print(df[i].value_counts())
# as we see that there is extra categories which we have to convert it into No.
df.MultipleLines = df.MultipleLines.replace('No phone service','No')
df.OnlineSecurity = df.OnlineSecurity.replace('No internet service','No')
df.OnlineBackup = df.OnlineBackup.replace('No internet service','No')
df.DeviceProtection = df.DeviceProtection.replace('No internet service','No')
df.TechSupport = df.TechSupport.replace('No internet service','No')
df.StreamingTV = df.StreamingTV.replace('No internet service','No')
df.StreamingMovies = df.StreamingMovies.replace('No internet service','No')


# In[15]:


# we have to handel this all categorical variables
# there are mainly Yes/No features in most of the columns
# we will convert Yes = 1 and No = 0
for i in cat:
    df[i] = df[i].replace('Yes',1)
    df[i] = df[i].replace('No',0)


# In[16]:


df.gender = df.gender.replace('Male',1)
df.gender = df.gender.replace('Female',0)


# In[26]:


df = df.drop('InternetService',axis=1)


# In[27]:


df = df.drop('Contract',axis=1)


# In[28]:


df = df.drop('PaymentMethod',axis=1)


# In[29]:


scale_cols = ['tenure','MonthlyCharges','TotalCharges']
# now we scling all the data 
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
df[scale_cols] = scale.fit_transform(df[scale_cols])


# In[30]:


df


# In[31]:


# independent and dependent variables
x = df.drop('Churn',axis=1)
y = df['Churn']


# In[32]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=10)
print(xtrain.shape)
print(xtest.shape)


# # Building ANN for Customer Churn Data

# In[33]:


# import tensorflow
import tensorflow as tf
#import keras 
from tensorflow import keras


# In[37]:


# define sequential model
model = keras.Sequential([
    # input layer
    keras.layers.Dense(16, input_shape=(16,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(10,activation = 'relu'),
    # we use sigmoid for binary output
    # output layer
    keras.layers.Dense(1, activation='sigmoid')
]
)


# In[38]:


# time for compilation of neural net.
model.compile(optimizer = 'adam',
             loss = 'binary_crossentropy',
             metrics = ['accuracy'])
# now we fit our model to training data
model.fit(xtrain,ytrain,epochs=10)


# In[39]:


# evalute the model
model.evaluate(xtest,ytest)


# In[40]:


# predict the churn values
ypred = model.predict(xtest)
print(ypred)
# unscaling the ypred values 
ypred_lis = []
for i in ypred:
    if i>0.5:
        ypred_lis.append(1)
    else:
        ypred_lis.append(0)
print(ypred_lis)


# # prediction

# In[41]:


#make dataframe for comparing the orignal and predict values
data = {'orignal_churn':ytest, 'predicted_churn':ypred_lis}
df_check = pd.DataFrame(data)
df_check.head(10)


# # performance matrix

# In[42]:


# checking for performance metrices
#importing classification_report and confusion metrics
from sklearn.metrics import confusion_matrix, classification_report
#print classification_report
print(classification_report(ytest,ypred_lis))
# ploting the confusion metrix plot
conf_mat = tf.math.confusion_matrix(labels=ytest,predictions=ypred_lis)
plt.figure(figsize = (17,7))
sb.heatmap(conf_mat, annot=True,fmt='d')
plt.xlabel('Predicted_number')
plt.ylabel('True_number')

