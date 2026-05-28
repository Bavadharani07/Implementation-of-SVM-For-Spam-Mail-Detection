# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the program.

2.Import the required libraries such as pandas, sklearn, and chardet.

3.Detect the encoding format of the spam.csv file.

4.Load the dataset using pandas.

5.Separate the dataset into input (message text) and output (spam or ham label).

6.Split the data into training data and testing data using train_test_split.

7.Convert the text messages into numerical form using CountVectorizer.

8.Train the SVM (Support Vector Machine) model using the training data.

9.Use the trained model to predict whether messages are spam or not.

10.Calculate and display the accuracy of the model.
 

## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: Bavadharani S
RegisterNumber:  212224040046


import chardet

file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)

import pandas as pd
data = pd.read_csv('spam.csv', encoding='Windows-1252')

data.info()
print(data.isnull().sum())

x = data["v1"].values
y = data["v2"].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

print("Predicted values:")
print(y_pred)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)



```

## Output:

### ENCODING :

<img width="845" height="41" alt="image" src="https://github.com/user-attachments/assets/db31bf59-7961-4df9-a2e9-516e201d41df" />

### HEAD():

<img width="723" height="209" alt="image" src="https://github.com/user-attachments/assets/d523f15c-f550-4cd0-a5aa-d6b77fc7f4fb" />

### isnul().sum :

<img width="317" height="146" alt="image" src="https://github.com/user-attachments/assets/681995a7-b7f9-4dd9-a963-2fe4cbd14d52" />

### Prediction of Y :

<img width="879" height="94" alt="image" src="https://github.com/user-attachments/assets/eb909d3e-0fb9-46d4-9a5f-904b4548b6a9" />


### Accuracy :

<img width="285" height="42" alt="image" src="https://github.com/user-attachments/assets/270e0781-2eb9-4c24-bec0-436a3d9f8e3e" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
