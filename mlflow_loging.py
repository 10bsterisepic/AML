pip install mlflow
import logging
import time
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(filename='mnist_sklearn_log.log', level=logging.INFO,
                   format="%(asctime)s - %(levelname)s - %(message)s")
logging.info('Starting MNIST (sklearn) Training...')

logging.info("Downloading MNIST dataset...")
mnist=fetch_openml('mnist_784', version=1, as_frame=False)

X=mnist.data/255.0          #Normalize pixel values
y=mnist.target.astype(int)  #convert labels to integers

logging.info(f"Dataset loaded: X shape={X.shape}, y shape:{y.shape}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

C=0.1
max_iter=100

logging.info(f"Hyperparameters: C={C}, max_iter={max_iter}")
mlflow.set_experiment("MNIST_Sklearn_Tracking")

with mlflow.start_run():
    start_time=time.time()
    mlflow.log_param("C", C)
    mlflow.log_param("max_iter", max_iter)

    model = LogisticRegression(C=C, max_iter=max_iter, solver='lbfgs', multi_class='multinomial')
    model.fit(X_train, y_train)
    logging.info('Model training completed')

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)

    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info("Classification Report:\n"+report)

    mlflow.sklearn.log_model(model, 'mnist_sklearn_model')

    runtime = time.time()-start_time
    mlflow.log_metric('runtime_seconds', runtime)
    logging.info(f"Total runtime = {runtime:.2f}s")

results_df = pd.DataFrame([{
    "C":C,
    "max_iter":max_iter,
    "accuracy":accuracy,
    "runtime_seconds":runtime
}])

results_df.to_csv('mnist_sklearn_results.csv', index=False)

plt.figure(figsize=(5, 4))
plt.bar(["accuracy"], [accuracy])
plt.title("MNIST Sklearn Model Accuracy")
plt.ylabel('Value')
plt.savefig('mnist_sklearn_accuracy_plot.png')
plt.show()

print("\nTraining complete. Logs saved, results tracked in MLFlow UI.")
