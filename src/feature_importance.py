import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
pipeline = joblib.load("models/student_performance_model.pkl")

# Get preprocessing and model
preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

# Get feature names
feature_names = preprocessor.get_feature_names_out()

# Get importance
importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print("\nTop 15 Important Factors:")
print(importance_df.head(15).to_string(index=False))

# Plot top 10
top_features = importance_df.head(10).sort_values("Importance")

plt.figure(figsize=(10, 6))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top Factors Affecting Student Performance")
plt.tight_layout()

plt.savefig("feature_importance.png")
plt.show()