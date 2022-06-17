# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm:
1. Import the standard Libraries.
2. Assign x and y values.
3. Import train_test_split from sklearn.model_selection and assign its values.
4. Import count vectorizer and assign it to cv.
5. Using SVC predict y_pred and print it.
6. Find accuracy and print it.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Iniyan S
RegisterNumber:  212220040053
*/

import pandas as pd
data = pd.read_csv("/content/sample_data/spam.csv",encoding = 'latin-1')

data.head()

data.info()

x = data["v1"].values

y = data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
### head() :
![OP](/OP1.png)
### info() :
![OP](/OP2.png)
### y_pred :
![OP](/OP3.png)
### accuracy :
![OP](/OP4.png)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
