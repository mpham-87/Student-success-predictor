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
