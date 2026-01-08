# Credit Risk Prediction - Assignment Explanation

## Objective
The objective of this task is to predict whether a loan applicant is likely to default on a loan (or have their loan approved) using the **Loan Prediction Dataset**. This involves data preprocessing, exploratory data analysis (EDA), model training, and evaluation.

---

## 1 Import Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
```

**Explanation:**
- `pandas` and `numpy` → For data manipulation and numerical operations.
- `matplotlib.pyplot` and `seaborn` → For visualization.
- `sklearn.model_selection.train_test_split` → Split data into training and testing sets.
- `LabelEncoder` → Convert categorical variables to numeric values for ML models.
- `LogisticRegression` and `DecisionTreeClassifier` → Two classification models used for prediction.
- `accuracy_score`, `confusion_matrix`, `ConfusionMatrixDisplay` → Evaluate model performance.

---

## 2 Load the Dataset

```python
df = pd.read_csv("loan_prediction.csv")  # adjust filename if needed
df.head()
```

**Explanation:**
- Loads the dataset into a Pandas DataFrame called `df`.
- `.head()` shows the first 5 rows for an initial look at the data.

---

## 3 Understand the Data

```python
df.info()
df.isnull().sum()
```

**Explanation:**
- `df.info()` → Displays column names, data types, and number of non-null entries.
- `df.isnull().sum()` → Counts missing values per column, which need handling before modeling.

---

## 4 Handle Missing Values

```python
# Numerical columns → Fill missing values with median
num_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Categorical columns → Fill missing values with mode
cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed']
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)
```

**Explanation:**
- Missing numerical values are replaced with **median** (robust to outliers).
- Missing categorical values are replaced with **mode** (most frequent value).
-  This ensures no null values remain before model training.

---

## 5 Exploratory Data Analysis (EDA)

### a) Loan Amount Distribution

```python
plt.hist(df['LoanAmount'], bins=20, edgecolor='black')
plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")
plt.show()
```

- Histogram shows the distribution of loan amounts among applicants.

### b) Education vs Loan Status

```python
sns.countplot(x='Education', hue='Loan_Status', data=df)
plt.title("Education vs Loan Status")
plt.show()
```

- Count plot visualizes how education level affects loan approval (`Loan_Status`).

### c) Applicant Income Distribution

```python
plt.hist(df['ApplicantIncome'], bins=20, edgecolor='black')
plt.title("Applicant Income Distribution")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()
```

- Histogram displays how applicant incomes are distributed in the dataset.

---

## 6 Encode Categorical Variables

```python
le = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])
```

**Explanation:**
- Converts categorical columns like `'Gender'`, `'Married'`, `'Education'` into numerical codes.
- Machine learning models require numeric inputs.

---

## 7 Split Data into Features and Target

```python
X = df.drop('Loan_Status', axis=1)  # Features
y = df['Loan_Status']               # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Explanation:**
- `X` contains all input features, `y` is the target (`Loan_Status`).
- Data is split into **training** (80%) and **testing** (20%) sets.
- `random_state=42` ensures reproducibility.

---

## 8 Train Classification Models

### Logistic Regression

```python
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
```

- Logistic Regression is a simple, widely-used classifier for binary outcomes.
- `max_iter=1000` ensures convergence.

### Decision Tree

```python
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
```

- Decision Tree splits the data based on feature values to predict the target.
- `random_state=42` ensures consistent tree structure.

---

## 9 Model Evaluation

### Accuracy

```python
acc_lr = accuracy_score(y_test, y_pred_lr)
acc_dt = accuracy_score(y_test, y_pred_dt)

print("Logistic Regression Accuracy:", acc_lr)
print("Decision Tree Accuracy:", acc_dt)
```

- `accuracy_score` measures how many predictions were correct.

### Confusion Matrix (Logistic Regression)

```python
cm = confusion_matrix(y_test, y_pred_lr)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.show()
```

- Confusion matrix shows:
  - True Positives (correctly predicted approvals)
  - True Negatives (correctly predicted rejections)
  - False Positives / Negatives (errors)

---

## 10 Conclusion

- Missing values were handled with median (numerical) and mode (categorical).
- Key features were visualized with histograms and countplots.
- Logistic Regression and Decision Tree models were trained.
- Model performance was evaluated using **accuracy** and **confusion matrix**.
- Logistic Regression provides stable performance for this dataset.

---

##  Skills Demonstrated

- Data cleaning and handling missing values
- Exploratory Data Analysis (EDA)
- Binary classification using machine learning models
- Model evaluation with accuracy and confusion matrix

