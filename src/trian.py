from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, GridSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
 
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "student-por.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
 
# Only information available before the final grade is used as input.
NUMERIC = ["studytime", "absences", "failures"]
CATEGORICAL = ["schoolsup", "famsup", "activities", "higher"]
FEATURES = NUMERIC + CATEGORICAL
 
raw = pd.read_csv(DATA, sep=";")
required = FEATURES + ["G3"]
missing_columns = [c for c in required if c not in raw.columns]
if missing_columns:
    raise ValueError(f"Missing expected columns: {missing_columns}")
 
# Remove records with no known final-grade label; leave feature gaps for imputation.
raw = raw.dropna(subset=["G3"]).copy()
X = raw[FEATURES].copy()
y = (raw["G3"] < 10).astype(int)  # 1 = may need support
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
 
numeric_steps = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
category_steps = Pipeline([
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocess = ColumnTransformer([
    ("num", numeric_steps, NUMERIC),
    ("cat", category_steps, CATEGORICAL),
])

# Fit on training records only; transform test records using the same rules.
train_array = preprocess.fit_transform(X_train)
test_array = preprocess.transform(X_test)
column_names = preprocess.get_feature_names_out()
for name, array, labels in [
    ("train", train_array, y_train),
    ("test", test_array, y_test),
]:
    cleaned = pd.DataFrame(array, columns=column_names)
    cleaned["needs_support"] = labels.to_numpy()
    cleaned.to_csv(OUT / f"preprocessed_{name}.csv", index=False)

# The model pipeline re-learns preprocessing within each CV training fold.
def make_model():
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            random_state=42, class_weight="balanced"
        )),
    ])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseline = make_model()
baseline_f1 = cross_val_score(
    baseline, X_train, y_train, cv=cv, scoring="f1"
)
print("Baseline cross-validation F1:", baseline_f1)
print("Baseline mean CV F1:", round(baseline_f1.mean(), 3))
 
search = GridSearchCV(
    estimator=make_model(),
    param_grid={
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_leaf": [1, 3],
    },
    scoring="f1", cv=cv, n_jobs=-1, refit=True,
)
search.fit(X_train, y_train)
print("Best settings:", search.best_params_)
print("Best mean CV F1:", round(search.best_score_, 3))

# Touch the held-out test set only after model selection is finished.
predictions = search.best_estimator_.predict(X_test)
metrics = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(precision_score(y_test, predictions, zero_division=0)),
    "recall": float(recall_score(y_test, predictions, zero_division=0)),
    "f1": float(f1_score(y_test, predictions, zero_division=0)),
    "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    "best_parameters": search.best_params_,
    "baseline_mean_cv_f1": float(baseline_f1.mean()),
    "best_mean_cv_f1": float(search.best_score_),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
}
print("Held-out test metrics:", json.dumps(metrics, indent=2))
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
joblib.dump(search.best_estimator_, OUT / "student_support_model.joblib")
print("Saved preprocessed CSVs, metrics and fitted model in outputs/")
