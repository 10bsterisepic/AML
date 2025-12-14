import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
print("Loading Heart Disease Dataset..")
df=fetch_openml(name = 'heart-disease', version=1, as_frame = True).frame
print('Dataset shape: ', df.shape)
print(df.head())

X = df.drop('target', axis=1)
y=df['target'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

models = {
    'GradientBoosting':GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric = 'logloss', use_label_encoder=False, random_state=42),
    'LightGBM': LGBMClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(verbose=False, random_state=42)
}

param_grids = {
    'GradientBoosting':{
        'n_estimators':[100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth':[3, 4]
    },
    'XGBoost':{
        'n_estimators':[100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth':[3, 4]
    },
    'LightGBM':{
        'n_estimators':[100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth':[3, 4]
    },
    'CatBoost':{
        'n_estimators':[100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth':[3, 4]
    }
}

def train_and_evaluate(name, model, X_train, X_test, y_train, y_test, param_grid):
    """Train model with hyperparameter tuning, evaluate performance."""
    print(f" \n{'='*20}\nTraining & Tuning {name}...\n{'='*20}" )
    start = time.time()

    grid=GridSearchCV( model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    runtime = time.time()-start

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    print(f"Best Params for {name}:{grid.best_params_}")
    print(f"{name}:Accuracy = {acc:.3f}, AUC={auc:.3f}, Runtime = {runtime:.2f}s")

    try:
        if hasattr(best_model, "feature_importances_"):
            importances = pd.Series(best_model.feature_importances_, index = X_train.columns)
            importances.sort_values().plot(kind='barh', figsize=(6, 5), title=f'{name}Feature Importance')
            plt.tight_layout()
            plt.show()
        elif name == "CatBoost":
            fi = best_model.get_feature_importance(prettified = True)
            plt.barh(fi['Feature Id'], fi['Importances'])
            plt.title(f'{name} Feature Importances')
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"Feature Importance not available for {name}:{e}")

    return{"Model":name, "Accuracy":acc, "AUC":auc, "Runtime(s)":runtime, "Best Params":grid.best_params_}

results = []
for name, model in models.items():
    param_grid = param_grids[name]
    res = train_and_evaluate(name, model, X_train, X_test, y_train, y_test, param_grid)
    results.append(res)

result_df = pd.DataFrame(results)
print("\n===Model Comparision Summary===")
print(result_df)

result_df.set_index('Model')[['Accuracy', 'AUC']].plot(kind='bar', figsize=(8, 4), title='Model Performance')
plt.ylabel("Score")
plt.show()

result_df.set_index('Model')['Runtime(s)'].plot(kind='bar', figsize=(8, 4), title='Runtime Comparision')
plt.ylabel("Seconds")
plt.show()
