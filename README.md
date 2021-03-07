# bank-notes_auth

Whenever you go to the bank to deposit some cash money, the cashier places banknotes in a machine which tells whether a banknote is real or not. This is a classification problem where we will be dealing here. We will be given a list of extracted features from the bank note and our task will be to classify it into legal or fraudulent note.

## To Run
To run the model locally, fist clone the repository using following command in your terminal.
```
git clone https://github.com/yashgupta417/bank-notes_auth.git
```
Then, open the `model.ipynb` file using jupyter notebook and press run.

## Dataset
Dataset has been taken from [here](https://archive.ics.uci.edu/ml/datasets/banknote+authentication#). The number of instances (rows) in the data set is 1372, and the number of variables (columns) is 5.
| f1 | f2 | f3 | f4 | label
|----|----|----|----|------
|3.931|1.8541|-0.023425|1.2314|0
|0.01727|8.693|1.3989|-3.9668|0
|3.2414|0.40971|1.4015|1.1952|0
|2.2504|3.5757|0.35273|0.2836|0
|-1.3971|3.3191|-1.3927|-1.9948|1
|0.39012|-0.14279|-0.031994|0.35084|1

In this code snippet, we are reading the dataset
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading data from .txt file
names=['f1','f2','f3','f4','label']
df=pd.read_csv('C:/Users/HP/Desktop/ML/bank-note-auth/data_banknote_authentication.txt',header=None,names=names)
df.head()
```

## Data pre-processing
In pre-processing step, we are standardizing the dataset using `StandardScaler` from sklearn.
```python
#standardization
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

#standardizing features
standardized_features=scaler.fit_transform(df.drop('label',axis=1))

#creating dataframe of standardized features
df_standardized=pd.DataFrame(standardized_features,columns=names[0:4])

#concatinating faetures and array
df_standardized=pd.concat([df_standardized,df['label']],axis=1)
df_standardized
```

## Data visualisation
Studying the distribution of features\
Feature 1 | Feature 2\
![f1](/f1.png)| ![f2](/f2.png)


Feature 3| Feature 4\
![f3](/f3.png)|![f4](/f4.png)

Plotting a pair plot to better understand the relationship between the features\
![Pair plot](/pair_plot.png)


## Training Model
Now, we are training `GaussianNB` model on the given dataset. We are training two models which differentiate on their prior probabilities. Model_1 is having `priors=[0.5,0.5]`
while on the other hand Model_2 is having `priors=[0.1,0.9]`.

Model_1:
```python
#training a gaussian naive bayes classifier

from sklearn.naive_bayes import GaussianNB
clf1=GaussianNB(priors=[0.5,0.5])
clf1.fit(X_train, Y_train)
```

Model_2:
```python
#training model with prior probablities [0.1,0.9]
clf2=GaussianNB(priors=[0.1,0.9])
clf2.fit(X_train, Y_train)
```
## Evaluating Model
Here we are evaluating model on various measures available in `sklearn.metrics`.
```python
#evalutating test dataset
Y_pred1=clf1.predict(X_test)

#accuracy
from sklearn.metrics import accuracy_score,classification_report
acc1=accuracy_score(Y_test, Y_pred1)
print("Accuracy of model1: ",acc1)

#classification report
report=classification_report(Y_test,Y_pred1)
print("\nClassification report: \n",report)

#confusion matrix
from sklearn.metrics import plot_confusion_matrix
cm=plot_confusion_matrix(clf1,X_test,Y_test)

cm.figure_.suptitle("Confusion Matrix")
plt.show()
```

Now plotting ROC curve
```python
#plotting ROC curve
plot_roc_curve(clf1, X_test, Y_test)
```

## Model Comparison
We trained two models, `clf1` and `clf2`. Now we will look at their results and compare them.

### Accuracy
Model_1: 0.84985\
Model_2: 0.79737

### Confusion Matrix
Model_1:\
![cm1](/cm1.png)\
Model_2:\
![cm2](/cm2.png)

### ROC
Model_1:\
![roc1](/roc1.png)\
Model_2:\
![roc2](/roc2.png)
