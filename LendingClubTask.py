# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:46:09 2020

@author: I519797
"""
import pandas as pd
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
# Load the data
df = pd.read_csv("LendingClubtrain.csv")
# Check the datatypes 
print(f"{df.dtypes}\n")
# Check if there is missing values print(f"Data types:\n{11 * '-'}")
print(f"Sum of null values in each feature:\n{35 * '-'}")
print(f"{df.isnull().sum()}")
#Luckily there are no null values
#View a preview of the first few rows of data
df.head()
#%% Do a heatmap visualization
sns.heatmap(df.corr())
#%%
# Get number of positve and negative examples
pos = df[df["loan_status"] == "Fully Paid"].shape[0]
neg = df[df["loan_status"] == "Charged Off"].shape[0]
print(f"Positive examples = {pos}")
print(f"Negative examples = {neg}")
print(f"Proportion of positive to negative examples = {(pos / neg) * 100:.2f}%")
#%%#Plot the counts of fully paid vs. Charged off
df['loan_status'].value_counts().plot(kind='barh')
#%%
# Create dummy variables from the variables which have string values
categoricalFeatures = ["loan_status","term", "grade", "sub_grade", "home_ownership", "verification_status", "purpose", "addr_state"]

# Iterate through the list of string categories and one hot encode them.
for feature in categoricalFeatures:
    onehot = pd.get_dummies(df[feature], prefix=feature)
    df = df.drop(feature, axis=1)
    df = df.join(onehot)
#%%
    # Liblinear is a solver that is effective for relatively smaller datasets.
lr = LogisticRegression(solver='liblinear', class_weight='balanced')
#need to define x-train and y-train - y is dependant variable and x is independant variables
# We will follow an 80-20 split pattern for our training and test data
#get rid of the column of charged off loans
df=df.drop(['loan_status_Charged Off'], axis=1)
#set the dependant variable y
y=df[['loan_status_Fully Paid']].values.ravel()
X_train,X_test,y_train,y_test = train_test_split(df, y, test_size=0.2, random_state = 0)
lr.fit(X_train, y_train)
#%%
# Make predictions on the test data
df2 = pd.read_csv("LendingClubtest.csv")
print(f"{df2.dtypes}\n")
print(f"Sum of null values in each feature:\n{35 * '-'}")
print(f"{df2.isnull().sum()}")

categoricalFeatures2 = ["term", "grade", "sub_grade", "home_ownership", "verification_status", "purpose", "addr_state"]
for feature in categoricalFeatures2:
    onehot = pd.get_dummies(df2[feature], prefix=feature)
    df2 = df2.drop(feature, axis=1)
    df2 = df2.join(onehot)
df2["loan_status_Fully Paid"] = 1
df2.head()
X_test=df2
#%%
y_pred = lr.predict_proba(X_test)

