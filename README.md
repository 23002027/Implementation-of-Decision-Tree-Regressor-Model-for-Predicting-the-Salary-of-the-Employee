# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries for data handling, preprocessing, modeling, and evaluation.
2. Load the dataset from the CSV file into a pandas DataFrame.
3. Check for null values and inspect data structure using .info() and .isnull().sum().
4. Encode the categorical "Position" column using LabelEncoder.
5. Split features (Position, Level) and target (Salary), then divide into training and test sets.
6. Train a DecisionTreeRegressor model on the training data.
7. Predict on test data, calculate mean squared error and R² score, and make a sample prediction.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by  : KAMESH RR 
RegisterNumber: 212223230095 
*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
data = pd.read_csv("/content/Salary.csv")
data.head()
```
```
data.info()
```
```
data.isnull().sum()
```
```
le=LabelEncoder()
data['Position']=le.fit_transform(data['Position'])
data.head()
```
```
# defining x and y and splitting them
x = data[["Position", "Level"]]
y = data["Salary"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
# predicting test values with model
y_pred = dt.predict(x_test)
mse = metrics.mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error : {mse}")
```
```
r2=metrics.r2_score(y_test,y_pred)
print(f"R2 Score : {r2}")
```
```
dt.predict(pd.DataFrame([[5,6]], columns=["Position", "Level"]))
```
## Output:

**Head Values**

![image](https://github.com/user-attachments/assets/e57307e5-4e40-4766-87ac-24b74fa4b186)

**DataFrame Info**

![image](https://github.com/user-attachments/assets/b4a3ef2c-e80a-4aa5-a464-5967f5e3a1a5)

**Sum - Null Values**

![image](https://github.com/user-attachments/assets/73f0f144-60a3-4c87-a3cb-e996064e7d91)

**Encoding position feature**

![image](https://github.com/user-attachments/assets/291dfa96-497b-4daf-b736-0cc40b3afef8)

**Mean Squared Error**

![image](https://github.com/user-attachments/assets/a31100c4-0ddb-4b15-aca1-3cf6a7db54fd)

**R2 Score**

![image](https://github.com/user-attachments/assets/cd8c1bbe-b00c-4493-b401-670fd47a2907)

**Final Prediction on Untrained Data**

![image](https://github.com/user-attachments/assets/f94ab572-8838-459c-82a9-8e14e742b444)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
