# bank-notes_auth

Whenever you go to the bank to deposit some cash money, the cashier places banknotes in a machine which tells whether a banknote is real or not. This is a classification problem where we will be dealing here. We will be given a list of extracted features from the bank note and our task will be to classify it into legal or fraudulent note.

## Dataset
The number of instances (rows) in the data set is 1372, and the number of variables (columns) is 5.
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
#reading data from .txt file
names=['f1','f2','f3','f4','label']
df=pd.read_csv('C:/Users/HP/Desktop/ML/bank-note-auth/data_banknote_authentication.txt',header=None,names=names)
df.head()
```

## Data pre-processing
Here we are standardizing the dataset.
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
Studying the distribution of features\\
Feature 1\
![f1](/f1.png)\
Feature 2\
![f2](/f2.png)\
Feature 3\
![f3](/f3.png)\
Feature 4\
![f4](/f4.png)\

Plotting a pair plot to better understand the relationship between the features\
![Pair plot](/pair_plot.png)\
