# 04a_train_individual_alertness_models.py
import pandas as pd
import numpy as np
import joblib
import json
import sys
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

try:
    from config import (FEATURES_DIR, MODELS_DIR, TEST_SIZE, RANDOM_STATE, CV_FOLDS)
    from utils import clean_and_recreate_dirs, log_and_save_df
except ImportError:
    print("Error: config.py or utils.py not found.")
    sys.exit(1)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def train_and_evaluate(X_train, y_train, X_test, y_test, model, param_grid, model_name):
    """Tunes and evaluates a single model, returning its best parameters."""
    print(f"\n--- Tuning {model_name} ---")
    cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1_weighted', cv=cv_strategy, n_jobs=1,
                               verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    return best_model, best_cv_score, test_accuracy, test_f1_weighted, best_params


def main():
    clean_and_recreate_dirs([MODELS_DIR])
    print("--- Starting Classic Model Hyperparameter Search ---")

    feature_files = {
        "MFCC_Delta": "alertness_mfcc_delta_features.csv",
        "Log-Mel": "alertness_log_mel_features.csv"
    }

    models_to_train = {
        "SVM": (SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced'),
                {'C': [1, 10, 100], 'gamma': ['scale', 'auto']}),
        "RF": (RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced_subsample'),
               {'n_estimators': [200, 300], 'max_depth': [10, 20]}),
        "KNN": (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']})
    }
    if XGB_AVAILABLE:
        models_to_train["XGBoost"] = (XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss', use_label_encoder=False, n_jobs=1 ),
                                      {'n_estimators': [100, 200], 'max_depth': [3, 5]})

    all_results = []
    best_hyperparams = {}

    for feature_name, filename in feature_files.items():
        print(f"\n{'=' * 20} PROCESSING: {feature_name} {'=' * 20}")
        best_hyperparams[feature_name] = {}
        df = pd.read_csv(FEATURES_DIR / filename)
        X = df.drop(columns=['filename', 'label'])
        y = df['label']
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=TEST_SIZE,
                                                            random_state=RANDOM_STATE, stratify=y_encoded)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        dataset_path = MODELS_DIR / f"dataset_{feature_name.lower()}.npz"
        np.savez(dataset_path, X_train_scaled=X_train_scaled, y_train=y_train, X_test_scaled=X_test_scaled,
                 y_test=y_test)
        joblib.dump(scaler, MODELS_DIR / f"scaler_{feature_name.lower()}.joblib")
        joblib.dump(le, MODELS_DIR / f"label_encoder_{feature_name.lower()}.joblib")

        for model_name, (model, param_grid) in models_to_train.items():
            _, cv_f1, test_acc, test_f1, params = train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test,
                                                                     model, param_grid, model_name)
            best_hyperparams[feature_name][model_name] = params
            all_results.append(
                {'Feature Set': feature_name, 'Model': model_name, 'CV F1-W': cv_f1, 'Test Acc': test_acc,
                 'Test F1-W': test_f1})

    params_path = MODELS_DIR / "best_hyperparameters.json"
    with open(params_path, 'w') as f:
        json.dump(best_hyperparams, f, indent=4)
    print(f"\nSaved best hyperparameters to {params_path}")

    if all_results:
        log_and_save_df("Classic Model Performance", pd.DataFrame(all_results), include_index=False)
    print("\n--- Classic Model Search Complete ---")


if __name__ == "__main__":
    main()