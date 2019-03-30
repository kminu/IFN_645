import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import pydot
from io import StringIO
from sklearn.tree import export_graphviz
from tools import data_prep 

#%matplotlib inline

df = data_prep()
y = df['IsBadBuy']
X = df.drop(['IsBadBuy'], axis=1)
df.count()

# Splitting the dataset
rs = 10
X_mat = X.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.30, stratify=y, random_state=rs)

# Model construction and evaluation
# simple decision tree training
model = DecisionTreeClassifier(random_state=rs)
model.fit(X_train, y_train)

# decision tree evaluation
print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

# Confusion Matrix
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# grab feature importances from the model and feature name from the original X
importances = model.feature_importances_
feature_names = X.columns

indices = np.argsort(importances)
indices = np.flip(indices, axis=0)

# limit to 20 features, you can leave this out to print out everything
indices = indices[:20]

print("\n\n*********** Feature Importances ************\n")   
for i in indices:
    print(f"{feature_names[i]:<35}:{importances[i]}")
    

# Another Decition Tree model
#retrain with a small max_depth limit 3
model_2 = DecisionTreeClassifier(max_depth=3, random_state=rs, )
model_2.fit(X_train, y_train)

print("Train accuracy:", model_2.score(X_train, y_train))
print("Test accuracy:", model_2.score(X_test, y_test))

y_pred = model_2.predict(X_test)
print(classification_report(y_test, y_pred))

#retrain with a small max_depth limit 4
model_3 = DecisionTreeClassifier(max_depth=4, random_state=rs, criterion='entropy',)
model_3.fit(X_train, y_train)

print("Train accuracy:", model_3.score(X_train, y_train))
print("Test accuracy:", model_3.score(X_test, y_test))

y_pred = model_3.predict(X_test)
print(classification_report(y_test, y_pred))

#max_depth 4.8 maximal optimum

# grab feature importances from the model and feature name from the original X
importances = model_3.feature_importances_
feature_names = X.columns

indices = np.argsort(importances)
indices = np.flip(indices, axis=0)

# limit to 20 features, you can leave this out to print out everything
indices = indices[:20]


print("\n\n*********** Feature Importances ************\n")   
for i in indices:
    print(f"{feature_names[i]:<35}:{importances[i]}")
# another model with params change
# retrain with a small max_depth limit 4
model_3 = DecisionTreeClassifier(max_depth=4, random_state=rs, criterion = 'gini', splitter='best'  )
model_3.fit(X_train, y_train)

print("Train accuracy:", model_3.score(X_train, y_train))
print("Test accuracy:", model_3.score(X_test, y_test))

y_pred = model_3.predict(X_test)
print(classification_report(y_test, y_pred))

#max_depth 4 maximal optimum

# grab feature importances from the model and feature name from the original X
importances = model_3.feature_importances_
feature_names = X.columns

indices = np.argsort(importances)
indices = np.flip(indices, axis=0)

# limit to 20 features, you can leave this out to print out everything
indices = indices[:20]


print("\n\n***********  Feature Importances ************\n")   
for i in indices:
    print(f"{feature_names[i]:<35}:{importances[i]}")


# visualize with graphViz
# refer to png file in your folder
dotfile = StringIO()
export_graphviz(model_3, out_file=dotfile, feature_names=X.columns)
graph = pydot.graph_from_dot_data(dotfile.getvalue())
graph[0].write_png("model_3Tree.png") # saved in the following file - will return True if successful


test_score = []
train_score = []

# check the model performance for max depth from 2-20
for max_depth in range(2, 21):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=rs)
    model.fit(X_train, y_train)
    
    test_score.append(model.score(X_test, y_test))
    train_score.append(model.score(X_train, y_train))


plt.plot(range(2, 21), train_score, 'b', range(2,21), test_score, 'r', )
plt.xlabel('max_depth\nBlue = training acc. Red = test acc.')
plt.ylabel('accuracy')

# grid search CV
params = {'criterion': ['gini', 'entropy'],
          'max_depth': range(2, 6),
          'min_samples_leaf': range(20, 60, 10)}

cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(random_state=rs), cv=10)
cv.fit(X_train, y_train)

print("Train accuracy:", cv.score(X_train, y_train))
print("Test accuracy:", cv.score(X_test, y_test))

# test the best model
y_pred = cv.predict(X_test)
print(classification_report(y_test, y_pred))

# print parameters of the best model
print(cv.best_params_)

# grid search CV #2
params = {'criterion': ['gini', 'entropy'],
          'max_depth': range(2, 6),
          'min_samples_leaf': range(20, 35)}

cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(random_state=rs), cv=10)
cv.fit(X_train, y_train)

print("Train accuracy:", cv.score(X_train, y_train))
print("Test accuracy:", cv.score(X_test, y_test))

# test the best model
y_pred = cv.predict(X_test)
print(classification_report(y_test, y_pred))

# print parameters of the best model
print(cv.best_params_)

# At max_depth 5 ovefitting is  evident
    
def analyse_feature_importance(dm_model, feature_names, n_to_display=20):
    # grab feature importances from the model
    importances = dm_model.feature_importances_
    
    # sort them out in descending order
    indices = np.argsort(importances)
    indices = np.flip(indices, axis=0)

    # limit to 20 features, you can leave this out to print out everything
    indices = indices[:n_to_display]

    print("\n\n*********** Feature Importances ************\n")   
    for i in indices:
        print(f"{feature_names[i]:<35}:{importances[i]}")

def visualize_decision_tree(dm_model, feature_names, save_name):
    dotfile = StringIO()
    export_graphviz(dm_model, out_file=dotfile, feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dotfile.getvalue())
    graph[0].write_png(save_name) # saved in the following file
    
analyse_feature_importance(cv.best_estimator_, X.columns, 20)
visualize_decision_tree(cv.best_estimator_, X.columns, "optimal_tree.png")