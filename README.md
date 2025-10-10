# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries

2.Prepare the dataset (Hours Studied vs Marks Scored)

3.Convert data into a DataFrame

4.Split the dataset into training and testing sets

5.Create and train the Linear Regression model

6.Predict marks using the test data

7.Evaluate the model’s performance

8.Visualize the regression line and data points

## Program:
```
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset: Hours Studied vs Marks Scored
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Marks_Scored': [20, 22, 34, 38, 46, 52, 58, 62, 70, 78]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['Hours_Studied']]   # Feature must be 2D
y = df['Marks_Scored']

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Print coefficients
print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

# Evaluate the model
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Plotting the regression line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.title('Simple Linear Regression: Marks vs Hours')
plt.legend()
plt.grid(True)
plt.show()

```

## Output:
<img width="872" height="797" alt="image" src="https://github.com/user-attachments/assets/a4f8738e-9cf4-4f20-8b60-455eabd1ea75" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
