import pandas as pd

# Load dataset
data = pd.read_csv("data/StudentPerformanceFactors.csv")

# -----------------------------
# Handle Missing Values
# -----------------------------

data["Teacher_Quality"] = data["Teacher_Quality"].fillna(
    data["Teacher_Quality"].mode()[0]
)

data["Parental_Education_Level"] = data[
    "Parental_Education_Level"
].fillna(
    data["Parental_Education_Level"].mode()[0]
)

data["Distance_from_Home"] = data[
    "Distance_from_Home"
].fillna(
    data["Distance_from_Home"].mode()[0]
)

# -----------------------------
# Create Target Class
# -----------------------------

def classify(score):
    if score >= 70:
        return "High"
    elif score >= 65:
        return "Average"
    else:
        return "Low"

data["Performance"] = data["Exam_Score"].apply(classify)

# -----------------------------
# Save Processed Data
# -----------------------------

data.to_csv(
    "data/student_processed.csv",
    index=False
)

print("Processed dataset saved!")

print("\nClass Distribution:")
print(data["Performance"].value_counts())

print("\nMissing Values:")
print(data.isnull().sum())