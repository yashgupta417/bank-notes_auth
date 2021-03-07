# bank-notes_auth

Whenever you go to the bank to deposit some cash money, the cashier places banknotes in a machine which tells whether a banknote is real or not. This is a classification problem where we will be dealing here. We will be given a list of extracted features from the bank note and our task will be to classify it into legal or fraudulent note.

## Dataset
The number of instances (rows) in the data set is 1372, and the number of variables (columns) is 5.

In this code snippet, we are reading the dataset
```python
#reading data from .txt file
names=['f1','f2','f3','f4','label']
df=pd.read_csv('C:/Users/HP/Desktop/ML/bank-note-auth/data_banknote_authentication.txt',header=None,names=names)
df.head()
```

