import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data/student_processed.csv")


# 1. Performance Class Distribution

data["Performance"].value_counts().reindex(
    ["Low", "Average", "High"]
).plot(kind="bar")

plt.title("Student Performance Distribution")
plt.xlabel("Performance")
plt.ylabel("Number of Students")
plt.tight_layout()

plt.savefig("performance_distribution.png")
plt.close()


# 2. Exam Score Distribution

plt.figure(figsize=(8, 5))

plt.hist(data["Exam_Score"], bins=15)

plt.title("Exam Score Distribution")
plt.xlabel("Exam Score")
plt.ylabel("Number of Students")

plt.tight_layout()
plt.savefig("exam_score_distribution.png")
plt.close()

print("Both graphs created successfully!")