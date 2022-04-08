import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
import csv


##data input#

X= np.genfromtxt('2018 skt x.csv', delimiter=',',encoding='utf-8-sig',
                 dtype=int)
y= np.genfromtxt('2018 skt y.csv', delimiter=',',encoding='utf-8-sig',
                 dtype=int)


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=0)
## Feature Scaling#
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

##random forest method(Regressor/Classifier)#
regressor = RandomForestClassifier(n_estimators=10)
regressor.fit(X_train, y_train)

##produce the y with trained machine#
y_pred = regressor.predict(X_test)

##measurement of model fit#
##these are for regressor#
print('Mean Absolute Error:',
      metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:',
      metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


##confution matrix#
##these are for classifications#
print(confusion_matrix(y_pred,y_test))
print(classification_report(y_pred,y_test))
print(accuracy_score(y_pred, y_test))


##import the feature names for graphs#
f_names=pd.read_csv("feature names.csv",header=None)
with open('feature names.csv', newline='') as f:
    reader = csv.reader(f)
    fnames = list(reader)

##provide feature importance for evaluation#
importances = regressor.feature_importances_
std = np.std([tree.feature_importances_ for tree in regressor.estimators_], axis=0)
forest_importances = pd.Series(importances, index=f_names)
print(forest_importances)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()

##feature plots exporting#
tree.export_graphviz(regressor[9],feature_names = fnames, out_file="tree.dot")
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(regressor[0],feature_names=fnames, filled = True)
fig.savefig('rf_individualtree.png')

###################testing part##################################
NX= np.genfromtxt('2019 skt x.csv', delimiter=',',encoding='utf-8-sig',
                 dtype=int)
NRy= np.genfromtxt('2019 skt y.csv', delimiter=',',encoding='utf-8-sig',
                 dtype=int)
NX = sc.fit_transform(NX)
Ny = regressor.predict(NX)
print('Mean Absolute Error:',
      metrics.mean_absolute_error(NRy, Ny))
print('Mean Squared Error:',
      metrics.mean_squared_error(NRy, Ny))
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(NRy, Ny)))

print(confusion_matrix(NRy,Ny))
print(classification_report(NRy,Ny))
print(accuracy_score(NRy, Ny))

importances = regressor.feature_importances_
Nstd = np.std([tree.feature_importances_ for tree in regressor.estimators_], axis=0)
forest_importances = pd.Series(importances, index=f_names)
print(forest_importances)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=Nstd, ax=ax)
ax.set_title("Feature importances")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()

