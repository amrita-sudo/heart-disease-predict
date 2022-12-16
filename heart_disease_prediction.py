# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


file_path = "heart.csv"
df = pd.read_csv(file_path)
# Split the training and testing data
X = df.drop(columns = 'target')
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42) 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

lg_clf_1 = LogisticRegression()
lg_clf_1.fit(X_train, y_train)
lg_clf_1.score(X_train, y_train)

# Predict the target values for the train set.
y_train_pred = lg_clf_1.predict(X_train)

print(f"{'Train Set'.upper()}\n{'-' * 75}\nConfusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))

print("\nClassification Report:")
print(classification_report(y_train, y_train_pred))

y_test_pred = lg_clf_1.predict(X_test)

print(f"{'Test Set'.upper()}\n{'-' * 75}\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report")
print(classification_report(y_test, y_test_pred))

#But this logistic regression model (refer to the object stored in the lg_clf_1 variable) is created using all the features(or independent variables). It is quite possible that not all features are of imporatance for the classification of the labels in the target column. Therefore, we still can improve the model by reducing the number of features to obtain higher f1-scores.
# Normalise the train and test data-frames using the standard normalisation method.
def standard_scaler(series):
  new_series = (series - series.mean()) / series.std()
  return new_series

norm_X_train = X_train.apply(standard_scaler, axis = 0)
norm_X_test = X_test.apply(standard_scaler, axis = 0)

norm_X_train.describe()
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

# Create the empty dictionary.
dict_rfe = {}

# Create a loop
for i in range(1, len(X_train.columns) + 1):
  lg_clf_2 = LogisticRegression()
  rfe = RFE(lg_clf_2, i) # 'i' is the number of features to be selected by RFE to fit a logistic regression model on norm_X_train and y_train.
  rfe.fit(norm_X_train, y_train)

  rfe_features = list(norm_X_train.columns[rfe.support_]) # A list of important features chosen by RFE.
  rfe_X_train = norm_X_train[rfe_features]
  
  # Build a logistic regression model using the features selected by RFE.
  lg_clf_3 = LogisticRegression()
  lg_clf_3.fit(rfe_X_train, y_train)
  
  # Predicting 'y' values only for the test set as generally, they are predicted quite accurately for the train set.
  y_test_pred = lg_clf_3.predict(norm_X_test[rfe_features])

  f1_scores_array = f1_score(y_test, y_test_pred, average = None)
  dict_rfe[i] = {"features": list(rfe_features), "f1_score": f1_scores_array} # 'i' is the number of features to be selected by RFE.
  pd.options.display.max_colwidth = 100 
f1_df = pd.DataFrame.from_dict(dict_rfe, orient = 'index')
f1_df
lg_clf_4 = LogisticRegression()
rfe = RFE(lg_clf_4, 3)

rfe.fit(norm_X_train, y_train)

rfe_features = norm_X_train.columns[rfe.support_]
print(rfe_features)
final_X_train = norm_X_train[rfe_features]
  
lg_clf_4 = LogisticRegression()
lg_clf_4.fit(final_X_train, y_train)
  
y_test_predict = lg_clf_4.predict(norm_X_test[rfe_features])
final_f1_scores_array = f1_score(y_test, y_test_predict, average = None)
print(final_f1_scores_array)