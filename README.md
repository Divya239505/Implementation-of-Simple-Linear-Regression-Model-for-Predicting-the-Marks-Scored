# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])


m = 0        
c = 0        
L = 0.01     
epochs = 1000  

n = float(len(X))  


for i in range(epochs):
    Y_pred = m * X + c  
    D_m = (-2/n) * sum(X * (Y - Y_pred))  
    D_c = (-2/n) * sum(Y - Y_pred)        
    m = m - L * D_m   
    c = c - L * D_c  

print(f"Final slope (m): {m}")
print(f"Final intercept (c): {c}")


Y_pred = m * X + c

plt.scatter(X, Y, color="red", label="Data Points")
plt.plot(X, Y_pred, color="blue", label="Best Fit Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Linear Regression using Gradient Descent")
plt.show()
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![alt text](<Screenshot 2025-10-04 161336.png>)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
