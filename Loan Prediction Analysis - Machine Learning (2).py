#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# <table>
#   <tr><td>
#     <img src="https://www.gannett-cdn.com/-mm-/9e1f6e2ee20f44aa1f3be4f71e9f3e52b6ae2c7e/c=0-110-2121-1303/local/-/media/2020/03/10/USATODAY/usatsports/getty-mortgage-house-calculator.jpg?width=2121&height=1193&fit=crop&format=pjpg&auto=webp"
#          width="400" height="600">
#       <tr><td align="center">
#   </td></tr>
#   </td></tr>
# </table>
# 
# The main goal of this project is to use Machine Learning techniques to determine whether a loan should be approved or not based on the past information of a person. This project includes:
# 
# 1. Data Cleaning
# 2. Data Visualizations
# 3. Transforming data
# 4. Identifying outliers 
# 5. Model Evaluations.
# 
# The libraries used in this project are:
# 1. sklearn
# 2. matplotlib
# 3. numpy
# 4. pandas
# 5. seaborn
# 
# 
# There are different models to train your data, here we will be using:
# 
# 1. logistic regression
# 2. decision trees
# 3. random forest
# 4. Hyperparameter Tuning method
# 
# 
# # Dataset
# 
# This dataset is named [Loan Prediction Dataset](https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset) data set. The dataset contains 613 records and attributes: Loan_ID, Gender, Married, Dependents, Education, Self_Employed, Applicant Income, Co-applicant Income, Loan Amount, Loan Amount Term, Credit History, Property Area   , and Loan_Status.
# 
# # Libraries

# In[ ]:


import os #paths to file
import numpy as np # linear algebra
import pandas as pd # data processing
import warnings# warning filter


#ploting libraries
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings# warning filter

#Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")


# # Load the datasets

# In[ ]:


#load the dataset
#training set
#download the data to run the report
tr_df = pd.read_csv(r"/Users/mirandacheng7/Downloads/Loan Prediction/train_u6lujuX_CVtuZ9i.csv")
#testing set
te_df= pd.read_csv(r"/Users/mirandacheng7/Downloads/Loan Prediction/test_Y3wMUE5_7gLdaTN.csv")


# # Processing the dataset

# ##### Take a look at the datesets
# 
# Training set:

# In[ ]:


#display the first 5 rows of the training set
tr_df.head()


# Testing set:

# In[ ]:


#display the first 5 rows of the testing set
te_df.head()


# Size of each data set:

# In[ ]:


#print the size of each dataset
print(f"training set (row, col): {tr_df.shape}\n\ntesting set (row, col): {te_df.shape}")


# # Data Cleaning

# ### Find the missing values

# In[ ]:


tr_df.isnull().sum()


# ### Fill the missing values
# As Gender, Married, credit_history, and self_employed are categorical data, we will replace the missing value with the most requent value. 

# In[ ]:


#filling the missing data with mode
null_cols = ['Credit_History', 'Self_Employed', 'LoanAmount','Dependents', 'Loan_Amount_Term', 'Gender', 'Married']

for col in null_cols:
    tr_df[col] = tr_df[col].fillna(tr_df[col].dropna().mode().values[0])   

tr_df.isnull().sum().sort_values


# In[ ]:


#check if there are any duplicates
tr_df.duplicated().any()


# In[ ]:


#remove the id column for both datasets as it's not needed
tr_df.drop('Loan_ID',axis=1,inplace=True)
te_df.drop('Loan_ID',axis=1,inplace=True)

#print the size of each dataset
print(f"training set (row, col): {tr_df.shape}\n\ntesting set (row, col): {te_df.shape}")


# ## Data visalization

# First, let's split data into categorical and numberical data.
# For categorical data, we want to show counts in each categorical bin using bars, for numberic data, we want to see the distribution. 

# In[ ]:


#categorical columns
cat = tr_df.select_dtypes('object').columns.to_list()

#numerical columns
num = tr_df.select_dtypes('number').columns.to_list()

#numberical data
loan_num =  tr_df[num]
#categorical df
loan_cat = tr_df[cat]


# In[ ]:


loan_cat


# In[ ]:


#display the counts of observations using bars for each categorical column
for i in loan_cat:
    plt.figure(figsize=(5,5))
    total = float(len(loan_cat[i]))
    ax = sns.countplot(loan_cat[i],palette='Blues')
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2,height + 3,'{:.1f}%'.format(height/total*100),ha="center")
    ax.set_title(i)
    plt.show()


# In[ ]:


#display the distribution of each numerical column
for i in loan_num:
    plt.hist(loan_num[i], bins=30)
    plt.title(i)
    plt.show()


# Display categorical data by Loan status.

# In[ ]:


for i in cat[:-1]: 
    plt.figure(figsize=(15,10))
    total = float(len(loan_cat[i]))
    plt.subplot(2,3,1)
    ax=sns.countplot(x=i ,hue='Loan_Status', data=tr_df ,palette='Blues')
    plt.xlabel(i, fontsize=14)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2,height + 3,'{:.1f}%'.format(height/total*100),ha="center")


# ## Encoding data to numeric

# change categorical data into numeric format

# In[ ]:


from sklearn.preprocessing import LabelEncoder
cols = ["Gender", "Married", "Education", "Self_Employed","Property_Area","Loan_Status"]
le = LabelEncoder()
for col in cols:
    tr_df[col] = le.fit_transform(tr_df[col].astype(str))


# In[ ]:


tr_df['Dependents'].value_counts()


# In[ ]:


# As 3+ in Dependents column has not been changed to numberic, so we should replace it manually
tr_df['Dependents'] = np.where((tr_df.Dependents == '3+'), 3, tr_df.Dependents)


# In[ ]:


#plotting the correlation matrix
plt.figure(figsize=(15,10))
sns.heatmap(tr_df.corr(), annot = True, cmap='BuPu')


# Train-Test Split

# In[ ]:


X= tr_df.drop(columns = ['Loan_Status'], axis = 1)
y = tr_df['Loan_Status']


# In[ ]:


#Split the data into train-test split:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

print(f"X_training set (row, col): {X_train.shape}\n\ny_train (row, col): {y_train.shape}\n\nX_test set (row, col): {X_test.shape}\n\ny_test set (row, col): {y_test.shape}")


# ## Logistic Regression
# Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. (Wikipedia)
# 
# <table>
#   <tr><td>
#     <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Exam_pass_logistic_curve.jpeg/400px-Exam_pass_logistic_curve.jpeg"
#          width="400" height="400">
#       <tr><td align="center">
#   </td></tr>
#   </td></tr>
# </table>

# In[ ]:


LR = LogisticRegression()
LR.fit(X_train, y_train)


y_predict = LR.predict(X_test)

print(classification_report(y_test, y_predict))

# print out the accuracy score
LR_SC = accuracy_score(y_predict,y_test)
print(f"{round(LR_SC*100,2)}% Accurate")


# ## Decision Tree
# 
# Decision Trees are constructed by splitting a data set based on different conditions and the goal is to create a model that predicts the value of a target variable by learning simple decisions inferred from the data features. 
# 
# <table>
#   <tr><td>
#     <img src="https://upload.wikimedia.org/wikipedia/commons/e/eb/Decision_Tree.jpg"
#          width="400" height="400">
#       <tr><td align="center">
#   </td></tr>
#   </td></tr>
# </table>
# 
# 
# 

# In[ ]:


DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)

y_predict = DT.predict(X_test)

#prediction summary
print(classification_report(y_test, y_predict))

# print out the accuracy score
DT_SC = accuracy_score(y_predict,y_test)
print(f"{round(DT_SC*100,2)}% Accurate")


# ## Random Forest
# 
# Random forest is an ensemble of decision trees that are trained with a combination of learning models to increase the overall results. Random forest builds multiple decisions trees and combines them together to get a better prediction.
# 
# <table>
#   <tr><td>
#     <img src="https://www.tibco.com/sites/tibco/files/media_entity/2021-05/random-forest-diagram.svg"
#          width="400" height="400">
#       <tr><td align="center">
#   </td></tr>
#   </td></tr>
# </table>
# 
# 
# 

# In[ ]:


RF = RandomForestClassifier()
RF.fit(X_train, y_train)

y_predict = RF.predict(X_test)

#  prediction Summary
print(classification_report(y_test, y_predict))

# print out accuracy score
RF_SC = accuracy_score(y_predict,y_test)
print(f"{round(RF_SC*100,2)}% Accurate")


# ## Hyperparameter tuning

# In[ ]:


RF_tuning = RandomForestClassifier(n_estimators= 70,min_samples_split=25,max_depth = 7, max_features= 1)
RF_tuning.fit(X_train, y_train)

y_predict = RF.predict(X_test)

#  prediction Summary
print(classification_report(y_test, y_predict))

# print out the accuracy score
RF_tuning_SC = accuracy_score(y_predict,y_test)
print(f"{round(RF_tuning_SC*100,2)}% Accurate")


# In[ ]:


score = [DT_SC,RF_SC,RF_tuning_SC,LR_SC]
Models = pd.DataFrame({
    'Model': ["Decision Tree","Random Forest","Hyperparameter Tunning", "Logistic Regression"],
    'Accuracy Score': score})
Models.sort_values(by='Accuracy Score', ascending=False)


# # Conclusions
# 
# 1. Loan Status is the most dependent on credit history as they have a high correlation ratio
# 2. The random forest after using hyperparameter tuning is the most accurate as modifying parameters could achieve an optimal model architecture
