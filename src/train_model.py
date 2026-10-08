import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------------
# Load Processed Dataset
# -----------------------------------

data = pd.read_csv("data/student_processed.csv")


# -----------------------------------
# Separate Features and Target
# -----------------------------------

X = data.drop(columns=["Exam_Score", "Performance"])
y = data["Performance"]


# -----------------------------------
# Identify Columns
# -----------------------------------

categorical_columns = X.select_dtypes(
    include=["str", "object"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    exclude=["str", "object"]
).columns.tolist()


# -----------------------------------
# Preprocessing
# -----------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])


# -----------------------------------
# Random Forest Model
# -----------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# -----------------------------------
# Complete Pipeline
# -----------------------------------

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# -----------------------------------
# Train-Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------------
# Train Model
# -----------------------------------

print("Training model...")

pipeline.fit(X_train, y_train)


# -----------------------------------
# Predictions
# -----------------------------------

y_pred = pipeline.predict(X_test)


# -----------------------------------
# Evaluation
# -----------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    y_pred,
    labels=["Low", "Average", "High"]
))


# -----------------------------------
# Save Model
# -----------------------------------

joblib.dump(
    pipeline,
    "models/student_performance_model.pkl"
)

print("\nModel saved successfully!")