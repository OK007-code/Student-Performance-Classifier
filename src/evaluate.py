import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data = pd.read_csv("data/student_processed.csv")
model = joblib.load("models/student_performance_model.pkl")

X = data.drop(columns=["Exam_Score", "Performance"])
y = data["Performance"]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

y_pred = model.predict(X_test)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Low", "Average", "High"]
)

print("Confusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Low", "Average", "High"]
)

display.plot()
plt.title("Student Performance Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()