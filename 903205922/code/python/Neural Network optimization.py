
# coding: utf-8

# In[38]:

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display, HTML
from sklearn.model_selection import GridSearchCV
InteractiveShell.ast_node_interactivity = "all"
import matplotlib.pyplot as plt


# In[34]:

df = pd.read_csv('human_resources_data.csv')
display(df.head())


# In[35]:

from sklearn.model_selection import train_test_split

# split up the training and test data before we start doing anything so we don't generate bias in the experiments
X_train, X_test, y_train, y_test = train_test_split(df.ix[:,:-1], df.ix[:,-1:], test_size=0.3, random_state=42)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

X_train.shape
X_test.shape
y_train.shape


# In[36]:

tuned_parameters = [{'hidden_layer_sizes': [(3, 2), (4, 2), (5, 2), (6, 2), (3, 1), (4, 1), (5, 1), (6, 1), (7, 2)]}]
ann_clf = GridSearchCV(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), tuned_parameters, cv=10)
ann_clf.fit(X_train, y_train)
ann_optimized = ann_clf.best_estimator_
print(ann_clf.best_params_)


# In[43]:

import time

ann_optimized.fit(X_train, y_train)
ann_optimized.score(X_test, y_test)

start_ann_fit = time.time()
ann_optimized.fit(X_train, y_train)
end_ann_fit = time.time()
ann_optimized.score(X_test, y_test)
end_ann_query = time.time()


# In[48]:

df_results = pd.DataFrame(columns=['Accuracy', 'Training Time', 'Testing Time'], index=['Backprop', 'RHC', 'SA', 'GA'])
df_results['Accuracy'] = [0.957, .762, .762, .762]
df_results['Training Time'] = [end_ann_fit - start_ann_fit, 38.109, 37.998, 958.469]
df_results['Testing Time'] = [end_ann_query - end_ann_fit, 0.070, 0.048, 0.049]

df_results['Accuracy'].plot(kind='bar', title='Accuracy')
plt.show()
df_results['Training Time'].plot(kind='bar', title='Training time')
plt.show()
df_results['Testing Time'].plot(kind='bar', title='Testing time')
plt.show()


# In[47]:

rhc_error = pd.read_csv('rhc_error.csv', header=None)
sa_error = pd.read_csv('sa_error.csv', header=None)
ga_error = pd.read_csv('ga_error.csv', header=None)

errors_df = pd.DataFrame(columns=['rhc', 'sa', 'ga'])
errors_df['rhc'] = rhc_error[0]
errors_df['sa'] = sa_error[0]
errors_df['ga'] = ga_error[0]

errors_df.plot(title='Randomized Optimization Errors')
plt.show()

