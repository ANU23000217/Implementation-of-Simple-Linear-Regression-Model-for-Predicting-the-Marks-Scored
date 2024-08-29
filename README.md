# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

### Step 1:
1.Import the standard Libraries.

### Step 2:
2.Set variables for assigning dataset values.

### Step 3:
3.Import linear regression from sklearn.

### Step 4:
4.Assign the points for representing in the graph.

### Step 5:
5.Predict the regression for marks by using the representation of the graph.

### Step 6:
6.Compare the graphs and hence we obtained the linear regression for the given datas. 
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ANU RADHA N
RegisterNumber:  212223230018
*/
```
```

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)



```

## Output:


![image](https://github.com/user-attachments/assets/05c7fd5a-df95-4d42-a709-ba3b847e2f03)


![image](https://github.com/user-attachments/assets/6de10895-f7bc-4ad9-938f-b50b6455a5ef)


![image](https://github.com/user-attachments/assets/f74f7f1b-88bb-4464-9bd2-0875021d36f9)


![image](https://github.com/user-attachments/assets/f08aa322-10bf-48b1-994b-3a8cd0be0a39)


![image](https://github.com/user-attachments/assets/fc41a86e-dac8-4d46-a8d3-085e674eae76)



![image](https://github.com/user-attachments/assets/d128264a-47a8-4425-9852-1fa5d20c902c)


![image](https://github.com/user-attachments/assets/25fda892-abb1-49f1-b5bf-8ef8faf625f0)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
