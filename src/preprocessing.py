from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

root = Path(__file__).resolve().parents[1]


def load_data():
    df = pd.read_csv(root / "data" / "student_mental_health.csv")
    target = "Mental_Health_Risk"

    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df[target]
    )

    X_train = train_df.drop(columns=["Student_ID", target])
    y_train = train_df[target].copy()
    X_test = test_df.drop(columns=["Student_ID", target])
    y_test = test_df[target].copy()

    return X_train, X_test, y_train, y_test


def build_preprocessor(X_train):
    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    binary_features = ["Part_Time_Job", "Previous_Mental_Health_History"]
    ordinal_features = ["Physical_Activity", "Diet_Quality"]
    nominal_features = ["Gender"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[["No", "Yes"], ["No", "Yes"]]))
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[
            ["Low", "Moderate", "High"],
            ["Poor", "Average", "Good"]
        ])),
        ("scaler", StandardScaler())
    ])

    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("binary", binary_pipeline, binary_features),
        ("ordinal", ordinal_pipeline, ordinal_features),
        ("nominal", nominal_pipeline, nominal_features)
    ])


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print("Training shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Class percentages:")
    print(y_train.value_counts(normalize=True) * 100)
