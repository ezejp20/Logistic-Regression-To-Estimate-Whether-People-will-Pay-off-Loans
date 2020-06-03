# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:46:09 2020

@author: I519797
"""
#I have decided to explore this problem using logistic regression since it is a classification problem with a binary dependant variable.
#Import the libraries which I will need
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
# Load the data
df = pd.read_csv("LendingClubtrain.csv")
# Check the datatypes 
print(f"{df.dtypes}\n")
# Check if there are any missing values
print(f"Sum of null values in each feature:\n{35 * '-'}")
print(f"{df.isnull().sum()}")
#Luckily there are no null values
#View a preview of the first few rows of data
df.head()
#%% Do a heatmap visualization of the correlations between variables
sns.heatmap(df.corr())
#%%
# Get the number of positve and negative examples (Fully Paid = pos, Charged Off=neg)
pos = df[df["loan_status"] == "Fully Paid"].shape[0]
neg = df[df["loan_status"] == "Charged Off"].shape[0]
print(f"Positive examples = {pos}")
print(f"Negative examples = {neg}")
print(f"Proportion of loans Charged Off to Fully Paid = {(neg / pos) * 100:.2f}%")
#%%#Plot the counts of fully paid vs. Charged off
df['loan_status'].value_counts().plot(kind='barh')
#%%
# Create dummy variables from the variables which have string values, they need to be changed to a binary format 
categoricalFeatures = ["loan_status","term", "grade", "sub_grade", "home_ownership", "verification_status", "purpose", "addr_state"]

# Iterate through the list of string categories and use one-hot encode to change them to separate columns with 1's and 0's.
for feature in categoricalFeatures:
    onehot = pd.get_dummies(df[feature], prefix=feature)
    df = df.drop(feature, axis=1)
    df = df.join(onehot)
#%%
#Liblinear solver: Accuracy: 0.6415, Precision: 0.8797953964194374, Recall: 0.6417910447761194, F1 Score: 0.7421790722761596
#With lbfgs solver Accuracy: 0.581, Precision: 0.8481012658227848, Recall: 0.5833333333333334, F1 Score: 0.6912306558585114
#With Newton-cg solver did not converge 
#sag: ran very slowly, Accuracy: 0.565, Precision: 0.8366788321167883, Recall: 0.570273631840796, F1 Score: 0.6782544378698225
#saga: Accuracy: 0.566, Precision: 0.8394495412844036, Recall: 0.5690298507462687, F1 Score: 0.6782802075611565
lr = LogisticRegression(solver='saga', class_weight='balanced', max_iter=10000)
#need to define x-train and y-train - y is dependant variable and x is independant variables
#follow an 80-20 split pattern for training and test data
#get rid of the column of charged off loans
df=df.drop(['loan_status_Charged Off'], axis=1)
#set the dependant variable y
y=df[['loan_status_Fully Paid']].values.ravel()
X_train,X_test,y_train,y_test = train_test_split(df, y, test_size=0.2, random_state = 0)
X_train=X_train.drop(["loan_status_Fully Paid"], axis=1)
X_test=X_test.drop(["loan_status_Fully Paid"], axis=1)
lr.fit(X_train, y_train)
#%%
# Compare this vector of predictions of the target vector to determine the model performance.
y_pred = lr.predict(X_test)
#%%
# Build the confusion matrix.
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
class_names=[list(df.columns)] 
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# The heatmap requires that dataframe is passed in as the argument
sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="YlGnBu", fmt="g")
# Configure the heatmap parameters
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
# Print out performance metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1 Score:",metrics.f1_score(y_test, y_pred))
#%%
# Make predictions on the other dataset
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
X_test2=df2
X_test2=X_test2.drop(["loan_status_Fully Paid"], axis=1)
#%%
y_pred2 = lr.predict_proba(X_test2)
y_pred3=np.delete(y_pred2,0,1)
predictions = pd.DataFrame({'Probability': y_pred3[:, 0]})
predictions.to_csv(r'C:\Users\I519797\OneDrive - SAP SE\Documents\Data Science\PredictionssLiblinear.csv')

