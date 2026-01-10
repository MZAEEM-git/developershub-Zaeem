
# Customer Churn Prediction (Bank Customers)

## Task 3 – Machine Learning Project

## 1. Objective
The objective of this project is to predict customer churn in a bank. Customer churn refers to customers who stop using the bank’s services. By building a classification model, we aim to identify customers who are likely to leave and understand the factors influencing churn.

## 2. Dataset Description
Dataset Name: Churn Modelling Dataset

Target Variable:
- Exited (1 = Customer left, 0 = Customer stayed)

Important Features:
- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary

## 3. Skills Demonstrated
- Data cleaning and preprocessing
- Categorical data encoding
- Supervised classification modeling
- Feature importance analysis

## 4. Import Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

## 5. Load the Dataset
```python
df = pd.read_csv("Churn_Modelling.csv")
df.head()
```

## 6. Data Cleaning
```python
df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)
df.isnull().sum()
```

## 7. Encoding Categorical Variables

### Gender Encoding
```python
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
```

### Geography Encoding
```python
df = pd.get_dummies(df, columns=["Geography"], drop_first=True)
```

## 8. Feature Selection
```python
X = df.drop("Exited", axis=1)
y = df["Exited"]
```

## 9. Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

## 10. Model Training
```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

## 11. Model Evaluation
```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

### Confusion Matrix
```python
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.show()
```

### Classification Report
```python
print(classification_report(y_test, y_pred))
```

## 12. Feature Importance
```python
feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
```

## 13. Key Insights
- Age and Balance are strong indicators of churn
- Active members are less likely to leave
- Geography and gender also influence churn

## 14. Conclusion
The Random Forest model effectively predicts customer churn and helps identify key factors that influence customer decisions. This model can assist banks in improving retention strategies.
