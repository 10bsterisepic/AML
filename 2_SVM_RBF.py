 import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import time

#load Iris dataset and use only two features for visualization ease
iris=datasets.load_iris()
X=iris.data[:, :2] #only first two features
y=iris.target

#Binary classification setup: class 0 and class 1 only for simplicity
is_class_01=y!=2
X=X[is_class_01]
y=y[is_class_01]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler=StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

kernels=['linear', 'poly', 'rbf']
results={}

for kernel in kernels:
    if kernel=='poly':
        clf=SVC(kernel=kernel, degree=3, gamma='scale', random_state=42)
    else:
        clf=SVC(kernel=kernel, gamma='scale', random_state=42)
    start_time=time.time()
    clf.fit(X_train_scaled, y_train)
    train_time=time.time() - start_time

    y_pred=clf.predict(X_test_scaled)
    acc=accuracy_score(y_test, y_pred)
    cm=confusion_matrix(y_test, y_pred)

    results[kernel] = {
        'model':clf,
        'train_time':train_time,
        'accuracy':acc,
        'confusion_matrix':cm
    }

    print(f"Kernel: {kernel}")
    print(f"Training time: {train_time:.4f} sec")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("-"*30)

def plot_decision_boundary(X, y, model, title):
    h=.02
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    z=model.predict(np.c_[xx.ravel(), yy.ravel()])
    z=z.reshape(xx.shape)

    plt.contourf(xx, yy, z, alpha=0.5)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolors='k', cmap=plt.cm.Set1)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(*scatter.legend_elements(), title='Classes')
    plt.show()

for kernel in kernels:
    plot_decision_boundary(X_test_scaled, y_test, results[kernel]['model'], f"SVM with {kernel} kernel decision boundary")
