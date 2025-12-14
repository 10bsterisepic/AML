import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)
X=np.random.rand(100, 1)*6-3
y=0.5*X**2+X+2+np.random.rand(100, 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

degrees=range(1, 16)
train_errors=[]
val_errors=[]

for degree in degrees:
    poly=PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly=poly.fit_transform(X_train)
    X_val_poly=poly.transform(X_val)
    model=LinearRegression()
    model.fit(X_train_poly, y_train)

    y_train_pred=model.predict(X_train_poly)
    y_test_pred=model.predict(X_val_poly)

    train_mse=mean_squared_error(y_train, y_train_pred)
    test_mse=mean_squared_error(y_val, y_test_pred)
    train_errors.append(train_mse)
    val_errors.append(test_mse)

plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, marker='o', label='Training Error')
plt.plot(degrees, val_errors, marker='o', label='Testing Error')
plt.yscale('log')
plt.xlabel('Model Complexity (Polynomial Degree)')
plt.ylabel('Mean Squared Error')
plt.title('Bias-Variance Tradeoff in Polynomial Regression')
#plt.axvline(x=np.argmin(val_errors)+1, color='r', linestyle='--', label='Optimal Degree')
plt.legend()
plt.grid()
plt.show()

optimal_degree=degrees[np.argmin(val_errors)]
print(f"Optimal model complexity: Degree {optimal_degree}")
print(f"Minimum validation error: {min(val_errors):.4f}")
