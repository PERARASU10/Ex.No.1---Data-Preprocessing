# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Step 1:
Importing the libraries.

Step 2:
Importing the dataset.

Step 3:
Taking care of missing data.

Step 4:
Encoding categorical data.

Step 5:
Normalizing the data.

Step 6:
Splitting the data into test and train.

Step 7:
End the program.

## PROGRAM:
DEVELOPED BY : PERARASU M
REG NO : 212222100033
```
import pandas as pd

df=pd.read_csv("/content/Churn_Modelling.csv")

df.head()

df.isnull().sum()

df.drop(["RowNumber","Age","Gender","Geography","Surname"],inplace=True,axis=1)

print(df)

x=df.iloc[:,:-1].values

y=df.iloc[:,-1].values

print(x)

print(y)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df1 = pd.DataFrame(scaler.fit_transform(df))

print(df1)

from sklearn.model_selection import train_test_split

xtrain,ytrain,xtest,ytest=train_test_split(x,y,test_size=0.2,random_state=2)

print(xtrain)

print(len(xtrain))

print(xtest)

print(len(xtest))

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

df1 = sc.fit_transform(df)

print(df1)
```
## OUTPUT:
df.head() :
![image](https://github.com/PERARASU10/Ex.No.1---Data-Preprocessing/assets/118348589/44808cd7-b50b-4838-9b53-b263fe077883)

df.isnull().sum() :
![image](https://github.com/PERARASU10/Ex.No.1---Data-Preprocessing/assets/118348589/a7f72887-ac25-4c8f-b25b-7ec3be36a510)

df value :
![image](https://github.com/PERARASU10/Ex.No.1---Data-Preprocessing/assets/118348589/0cf77bcf-4edc-488c-a079-a449d6732b4b)

VALUES OF INPUT AND OUTPUT DATA ON VAR X AND Y :
![image](https://github.com/PERARASU10/Ex.No.1---Data-Preprocessing/assets/118348589/3ce1ec55-ef57-47b1-a775-dce696bddfc4)
![image](https://github.com/PERARASU10/Ex.No.1---Data-Preprocessing/assets/118348589/7f6562f6-ce4e-47b2-a677-80ab24625a0d)

NORMALIZING DATA:
![image](https://github.com/PERARASU10/Ex.No.1---Data-Preprocessing/assets/118348589/f53a2e6c-8670-44ad-89f7-36fa42a6d6fa)

X_TRAIN AND Y_TRAIN VALUES :
![image](https://github.com/PERARASU10/Ex.No.1---Data-Preprocessing/assets/118348589/c278af04-e91a-412c-a788-f257c696f341)

X AND Y VALUES :
![image](https://github.com/PERARASU10/Ex.No.1---Data-Preprocessing/assets/118348589/1f7e5646-e024-4812-933e-edbea01baf35)

X_TEST AND Y_TEST VALUES :
![image](https://github.com/PERARASU10/Ex.No.1---Data-Preprocessing/assets/118348589/329d759f-15e0-4068-b27f-a4dedc47a38c)


## RESULT
Thus,the program to perform Data preprocessing in a data set downloaded from Kaggle is implemented successfully .
