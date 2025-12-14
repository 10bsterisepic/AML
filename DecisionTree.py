import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, validation_curve, train_test_split

X, y = load_iris(return_X_y=True)

param_grid={
    'max_depth':[None, 2, 3, 4, 5, 6, 7, 8],
    'min_samples_split':[2, 4, 6, 8, 10]}
dt=DecisionTreeClassifier(random_state=42)
grid_search=GridSearchCV(
    dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

param_range=np.arange(1, 15)
train_scores, test_scores=validation_curve(
    DecisionTreeClassifier(min_samples_split = grid_search.best_params_['min_samples_split'], random_state=42),
    X, y, param_name="max_depth", param_range=param_range, cv=5, scoring='accuracy')

train_mean=np.mean(train_scores, axis=1)
test_mean=np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(param_range, train_mean, label="Training score", color="blue", marker='o')
plt.plot(param_range, test_mean, label="Cross validation score", color='green', marker='o')
plt.title('Validation Curve for Decision Tree')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend(loc='best')
plt.show()
