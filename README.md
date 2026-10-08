# 📊 Student Performance Classifier

A machine learning project that predicts a student's academic performance as **High, Average, or Low** based on academic, demographic, and related factors.

The project uses the **UCI Student Performance Dataset** and a **Random Forest Classifier** to make predictions.

---

## 🎯 Project Objective

The objective of this project is to build a machine learning model that:

- Takes relevant student attributes as input
- Classifies student performance as **High, Average, or Low**
- Provides a **confidence score** for the prediction
- Shows the **factors contributing most to the prediction**
- Displays a **confusion matrix** for model evaluation
- Provides an interactive prediction interface using Streamlit

---

## ✨ Features

### 1. Performance Classification
The model classifies students into three categories:

- **High:** G3 ≥ 15
- **Average:** 10 ≤ G3 < 15
- **Low:** G3 < 10

Here, G3 represents the final grade in the original dataset and is used to create the target class.

### 2. Confidence Score
The application displays the model's prediction confidence using the predicted class probability.

### 3. Contributing Factors
The project uses Random Forest feature importance to show the factors that contribute most to the model's predictions.

### 4. Confusion Matrix
A confusion matrix is generated to evaluate the model's classification performance across the three classes.

### 5. Interactive Interface
A Streamlit interface allows users to enter student information and receive a prediction.

---

## 🛠️ Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**
- **Streamlit**
- **Joblib**

---

## 📂 Project Structure

```text
student-performance-classifier/
│
├── data/
│   └── student-mat.csv
│
├── models/
│   ├── student_performance_model.pkl
│   └── preprocessor.pkl
│
├── src/
│   ├── eda.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── app.py
├── requirements.txt
└── README.md

📊 Dataset

The project uses the Student Performance Dataset from the UCI Machine Learning Repository.

Source:
https://archive.ics.uci.edu/dataset/320/student+performance

The project uses the student-mat.csv dataset, which contains information about students' academic performance and related factors.

The dataset contains 395 student records and 33 original attributes, including the final grade G3.

For model training:

G3 is used to create the target performance class.
G3 itself is not used as an input feature.
The remaining 32 attributes are used as model inputs.
🧠 Model and Approach
1. Target Creation

The original final grade G3 is converted into three performance classes:

G3 >= 15       → High
10 <= G3 < 15  → Average
G3 < 10        → Low
2. Feature Preparation

The dataset contains both numerical and categorical features.

Categorical features are converted into numerical form using One-Hot Encoding.

Numerical features are passed through without scaling because the selected Random Forest model does not require feature scaling.

3. Train-Test Split

The dataset is divided into:

80% training data
20% testing data

A stratified split is used to maintain the class distribution between training and testing data.

4. Machine Learning Model

A Random Forest Classifier is used with:

n_estimators = 100
random_state = 42
5. Model Performance

The model achieved an accuracy of approximately:

82.28%

on the test dataset.

📈 Evaluation

The model is evaluated using a confusion matrix.

The confusion matrix obtained on the test data is:

              Predicted
              Low  Average  High

Actual Low     22     4       0
Actual Average  7    31       0
Actual High     0     3      12

The model correctly classified 65 out of 79 test samples.