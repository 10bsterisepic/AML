from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
import numpy as np

data=load_breast_cancer()
X, y = data.data, data.target

base_models = [
    ('lr', LogisticRegression(max_iter=5000)),
    ('knn', KNeighborsClassifier()),
    ('dtc', DecisionTreeClassifier()),
    ('svm', SVC(kernel='linear', probability=True)),
    ('nb', GaussianNB())
]

#define stacking ensemble
meta_model=LogisticRegression()
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

#models to compare(base models + stacking)
models=base_models+[('stacking', stacking_model)]

#cross-validate and compare models
cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models:
    scores=cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"{name}:Mean Accuracy={np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
