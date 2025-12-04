import shap
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

data = load_breast_cancer()
X=pd.DataFrame(data.data, columns = data.feature_names)
y=data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

sv=shap_values[:,:,1]

feature_importance = pd.DataFrame({'feature':X_test.columns,
                    'mean_abs_SHAP': abs(sv).mean(axis=0)}).sort_values('mean_abs_SHAP', ascending=False)
print(feature_importance)
