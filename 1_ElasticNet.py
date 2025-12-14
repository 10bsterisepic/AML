from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score


X, y=fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

models={'Linear Regression':LinearRegression(), 'Ridge':Ridge(), 'Lasso':Lasso(), 'ElasticNet':ElasticNet()}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred=model.predict(X_test_scaled)
    mse=mean_squared_error(y_test, y_pred)
    r2=r2_score(y_test, y_pred)
    print(f"{name}:MSE = {mse:.4f}, R2 = {r2:.2f}")
    print(f"{name}Non-zero coefficients: {(model.coef_ !=0).sum()}/{(len(model.coef_))}\n")

plt.figure(figsize=(10, 6))
for name, model in models.items():
    plt.plot(model.coef_, marker='o', label=name)
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.legend()
plt.title('Model Coefficient Comparision')
plt.show()
