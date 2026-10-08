import pandas as pd

# Load dataset
data = pd.read_csv("data/StudentPerformanceFactors.csv")

print("Dataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns.tolist())

print("\nFirst 5 Rows:")
print(data.head())

print("\nData Types:")
print(data.dtypes)

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Rows:")
print(data.duplicated().sum())

print("\nExam Score Statistics:")
print(data["Exam_Score"].describe())

print("\nUnique Exam Scores:")
print(sorted(data["Exam_Score"].unique()))