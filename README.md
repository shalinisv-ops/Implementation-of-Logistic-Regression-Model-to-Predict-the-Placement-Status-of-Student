# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Import Required Libraries

Step 2: Load the Dataset

Step 3: Copy Data & Drop Unwanted Columns

Step 4: Check Data Quality

Step 5: Encode Categorical Variables

Step 6: Define Features (X) and Target (y)

Step 7: Split into Training and Testing Sets

Step 8: Build and Train Logistic Regression Model

Step 9: Make Predictions

Step 10: Evaluate the Model

Step 11: Predict for a New Student

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SHALINI SV
Register Number:  212224020051
*/
```
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("Placement_Data.csv")
print("First 5 rows of the dataset:")
print(data.head())

data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis=1)

print("\nData after dropping 'sl_no' and 'salary':")
print(data1.head())

print("\nChecking for missing values (True = missing):")
print(data1.isnull().any())

print("\nNumber of duplicate rows:")
print(data1.duplicated().sum())

cat_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", 
            "degree_t", "workex", "specialisation", "status"]

le = LabelEncoder()

for col in cat_cols:
    data1[col] = le.fit_transform(data1[col])

print("\nData after Label Encoding:")
print(data1.head())

X = data1.iloc[:, :-1]
y = data1["status"]

print("\nFeatures (X) sample:")
print(X.head())

print("\nTarget (y) sample:")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print("\nTraining and testing shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

lr = LogisticRegression(solver="liblinear")
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("\nPredicted values (y_pred):")
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_student = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]

new_prediction = lr.predict(new_student)

print("\nPrediction for new student (0 = Not Placed, 1 = Placed):")
print(new_prediction[0])
```

## Output:
<img width="583" height="261" alt="image" src="https://github.com/user-attachments/assets/594eef50-b5ae-44b1-ab88-8da5cdecaa1c" />

<img width="639" height="259" alt="image" src="https://github.com/user-attachments/assets/775487d0-3fdd-438a-9797-0f33ade309df" />

<img width="359" height="323" alt="image" src="https://github.com/user-attachments/assets/c1c96c90-00e2-4a45-b792-3fb26f2522e4" />

<img width="629" height="264" alt="image" src="https://github.com/user-attachments/assets/7e8af85d-6e2c-4200-baf2-2102df3b77cf" />

<img width="673" height="402" alt="image" src="https://github.com/user-attachments/assets/8cf39eab-9578-40e8-8b73-19902e0027cd" />

<img width="636" height="82" alt="image" src="https://github.com/user-attachments/assets/4e4ac171-6d7d-4962-ba96-44f911c99009" />

<img width="483" height="219" alt="image" src="https://github.com/user-attachments/assets/9d217989-d472-4d94-8ca5-f79fc06c0b13" />

<img width="1371" height="84" alt="image" src="https://github.com/user-attachments/assets/e52eec42-2f67-408d-8432-374967ae747a" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
