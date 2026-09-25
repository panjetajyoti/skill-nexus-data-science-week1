import numpy as np
import pandas as pd

# 1. Dataset create karna
data = {
    "EmployeeID": [101, 102, 103, 104, 105, 106, 107],
    "Department": ["Sales", "IT", "IT", "HR", "Sales", "IT", "HR"],
    "Salary": [52000, 78000, 85000, 48000, 54000, 92000, 50000],
    "Experience_Years": [2, 5, 6, 1, 3, 8, 2],
    "Performance_Score": [85, 90, 78, 88, 92, 95, 80],
}

df = pd.DataFrame(data)

# 2. Basic Inspection
print("--- DataFrame Info ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

# 3. Missing values check
print("\n--- Missing Values Count ---")
print(df.isnull().sum())

# 4. Department-wise Aggregations
dept_summary = (
    df.groupby("Department")
    .agg(
        Total_Employees=("EmployeeID", "count"),
        Avg_Salary=("Salary", "mean"),
        Avg_Experience=("Experience_Years", "mean"),
    )
    .reset_index()
)

print("\n--- Department Summary ---")
print(dept_summary)

# 5. Outlier Detection using IQR on Salary
Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df["Salary"] < lower_bound) | (df["Salary"] > upper_bound)]
print("\n--- Outliers Detected ---")
print("None" if outliers.empty else outliers)
