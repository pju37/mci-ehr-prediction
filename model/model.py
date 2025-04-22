
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
import os

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.utils import resample
from joblib import parallel_backend

file_names = ['model_10x_0yr.csv', 'model_10x_1yr.csv', 'model_10x_3yr.csv', 'model_10x_5yr.csv']
model_years = [0, 1, 3, 5]
results_all = []
feature_importances_all = []
all_feature_names = set()

# ========== Preprocess and Save Features ==========
for file, year in zip(file_names, model_years):
    df = pd.read_csv(file)
    categorical_features = ['Sex', 'Race']
    numerical_features = [col for col in df.columns if col not in categorical_features + ['mci', 'person_id']]
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('var', VarianceThreshold())]), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    X = df.drop(columns=['mci', 'person_id'])
    Y = df['mci'].astype(int)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=42)

    with parallel_backend('threading', n_jobs=-1):
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

    np.save(f'X_train_{year}.npy', X_train)
    np.save(f'X_test_{year}.npy', X_test)
    np.save(f'Y_train_{year}.npy', Y_train)
    np.save(f'Y_test_{year}.npy', Y_test)

    feature_names = preprocessor.transformers_[0][1].get_feature_names_out(numerical_features).tolist() +                     preprocessor.transformers_[1][1].get_feature_names_out(categorical_features).tolist()
    all_feature_names.update(feature_names)

    with open(f'feature_names_{year}.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

# ========== Model Training and Evaluation ==========

def evaluate_model(model, X_test, Y_test, year, model_name):
    n_iterations = 300
    n_size = 1000
    results = []
    for _ in range(n_iterations):
        X_res, Y_res = resample(X_test, Y_test, n_samples=n_size)
        Y_pred = model.predict(X_res)
        y_score = model.predict_proba(X_res)[:, 1]
        results.append({
            'Model': model_name,
            'Model Year': year,
            'Balanced Accuracy': balanced_accuracy_score(Y_res, Y_pred),
            'Accuracy': accuracy_score(Y_res, Y_pred),
            'AUROC': roc_auc_score(Y_res, y_score),
            'AUPRC': average_precision_score(Y_res, y_score),
            'Precision': precision_score(Y_res, Y_pred),
            'Recall': recall_score(Y_res, Y_pred),
            'F1': f1_score(Y_res, Y_pred)
        })
    return results

for file, year in zip(file_names, model_years):
    X_train = np.load(f'X_train_{year}.npy', allow_pickle=True)
    X_test = np.load(f'X_test_{year}.npy', allow_pickle=True)
    Y_train = np.load(f'Y_train_{year}.npy', allow_pickle=True)
    Y_test = np.load(f'Y_test_{year}.npy', allow_pickle=True)

    with open(f'feature_names_{year}.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    
    param_grid = {
        'n_estimators': [200, 300], 
        'learning_rate': [0.05, 0.1],  
        'max_depth': [3, 5], 
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0], 
        'scale_pos_weight': [8, 10],  
        'min_child_weight': [3, 4]  
    }

    models = {

        'XGBoost': xgb.XGBClassifier(
            objective='binary:logistic', tree_method='gpu_hist', predictor='gpu_predictor',
            use_label_encoder=False, eval_metric='auc', random_state=42, n_jobs=-1),
        'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42, n_jobs=-1),
        'LASSO': LogisticRegression(penalty='l1', solver='saga', max_iter=1000, random_state=42)
    }

    
    for model_name, model in models.items():

            if model_name == "RandomForest":
                rf_grid = {
                    'n_estimators': [1000],
                    'max_depth': [3, 5, 7, 9],
                    'max_features': ['sqrt', 'log2'],
                    'class_weight': ['balanced']
                }
                skf = StratifiedKFold(n_splits=5)
                model = GridSearchCV(model, rf_grid, cv=skf, scoring='f1', n_jobs=-1)


            if model_name == "LASSO":
                lr_grid = {
                    'C': [0.01, 0.1, 1, 10],  # inverse regularization strength
                    'penalty': ['l1'],
                    'solver': ['saga'],
                    'class_weight': ['balanced']
                }
                skf = StratifiedKFold(n_splits=5)
                model = GridSearchCV(model, lr_grid, cv=skf, scoring='f1', n_jobs=-1)

        if model_name == "XGBoost":
            skf = StratifiedKFold(n_splits=5)
            model = GridSearchCV(model, param_grid, cv=skf, scoring='f1', n_jobs=-1)

        model.fit(X_train, Y_train)
        with open(f'{model_name}_model_{year}.pkl', 'wb') as f:
            pickle.dump(model, f)

        if model_name == 'XGBoost':
            feature_importance = model.feature_importances_
            df_fi = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance,
                'Model Year': year,
                'Model': model_name
            })
            feature_importances_all.append(df_fi)

        results_all.extend(evaluate_model(model, X_test, Y_test, year, model_name))

# Save Results
results_df = pd.DataFrame(results_all)
results_df.to_csv("model_results_df.csv", index=False)

# Feature Importances
feature_importances_df = pd.concat(feature_importances_all)
feature_importances_df.to_csv("feature_importances_df.csv", index=False)

# SHAP Summary
for year in model_years:
    X_train = np.load(f'X_train_{year}.npy', allow_pickle=True)
    with open(f'feature_names_{year}.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    with open(f'XGBoost_model_{year}.pkl', 'rb') as f:
        model = pickle.load(f)
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary Plot - XGBoost {year}yr")
    plt.tight_layout()
    plt.savefig(f"shap_summary_{year}yr.png")
    plt.clf()
