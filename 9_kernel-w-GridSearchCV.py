import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

iris=datasets.load_iris()
X=iris.data
y=iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

para_grid=[
    {'kernel':['linear'], 'C':[0.01, 0.1, 1, 10, 100]},
    {'kernel':['poly'], 'C':[0.01, 0.1, 1, 10], 'degree':[2,3,4]},
    {'kernel':['rbf'], 'C':[0.01, 0.1, 1, 10], 'gamma':['scale', 0.1, 0.01, 0.001]}
]

grid_search=GridSearchCV(estimator=SVC(), param_grid=para_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print('Best Parameter:', grid_search.best_params_)
print('Best Cross-Validation Accuracy: ', grid_search.best_score_)

best_model=grid_search.best_estimator_
y_pred=best_model.predict(X_test)
print('\nClassification Report: \n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n ', confusion_matrix(y_test, y_pred))
print('Test Accuracy: ', best_model.score(X_test, y_test))

import seaborn as sns
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Best SVM Model')
plt.show()
