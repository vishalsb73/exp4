# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("bmi.csv")
df.head()
```
<img width="283" height="199" alt="Screenshot 2025-10-16 224323" src="https://github.com/user-attachments/assets/8d3c64b2-0eca-4bbd-a837-2e44c63024bb" />

```
df.dropna()
```
<img width="300" height="404" alt="Screenshot 2025-10-16 224331" src="https://github.com/user-attachments/assets/075d457d-dbfd-4229-8b89-972e195bbbbb" />

```
df_null_sum=df.isnull().sum()
df_null_sum
```
<img width="146" height="120" alt="Screenshot 2025-10-16 224336" src="https://github.com/user-attachments/assets/eb35e0c5-9209-4c95-b7bb-c09afc3e444d" />

```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```
<img width="155" height="83" alt="Screenshot 2025-10-16 224341" src="https://github.com/user-attachments/assets/2d96b3f9-cd2a-461c-bb61-4be4abdf6b22" />

```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("bmi.csv")
df1.head()
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("bmi.csv")
df1.head()
```
<img width="282" height="212" alt="Screenshot 2025-10-16 224347" src="https://github.com/user-attachments/assets/622efca4-9a95-45f0-aa97-9adadade9757" />

```
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
<img width="323" height="344" alt="Screenshot 2025-10-16 224353" src="https://github.com/user-attachments/assets/d21c8c1c-f05d-403d-93c9-f09fdd49ddbb" />

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="325" height="352" alt="Screenshot 2025-10-16 224359" src="https://github.com/user-attachments/assets/0c721914-03f1-4038-a7fd-c878e63d49c3" />

```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("bmi.csv")
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
<img width="320" height="411" alt="Screenshot 2025-10-16 224405" src="https://github.com/user-attachments/assets/bb862b05-fdff-4b5f-80c6-a247a821ce75" />

```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
```
<img width="340" height="205" alt="Screenshot 2025-10-16 224410" src="https://github.com/user-attachments/assets/8bd277cf-7b00-4c68-a346-12a6a534d0a6" />

```
df=pd.read_csv("titanic_dataset.csv")
df.info()
```
<img width="418" height="379" alt="Screenshot 2025-10-16 224415" src="https://github.com/user-attachments/assets/d9d6eade-98d7-4c47-bc74-93b171e23730" />

```
df_null_sum=df.isnull().sum()
df_null_sum
```
<img width="221" height="265" alt="Screenshot 2025-10-16 224421" src="https://github.com/user-attachments/assets/613e94c1-1136-4132-8d58-1d82f7bd41ac" />

```
categorical_columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch','Ticket','Fare','Cabin','Embarked']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
<img width="1132" height="417" alt="Screenshot 2025-10-16 224430" src="https://github.com/user-attachments/assets/c5c6406a-1be8-49de-8c1c-ccac14661a08" />

```
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
<img width="745" height="400" alt="Screenshot 2025-10-16 224437" src="https://github.com/user-attachments/assets/0e630712-192a-42b6-a438-be62e6e6ba0a" />

```
X = df.drop(columns=['Survived'])
y = df['Survived']
df1=df.drop(columns=['Name','Sex','Ticket','Cabin','Embarked'])
df1['Age'].isnull().sum()
```
<img width="501" height="410" alt="Screenshot 2025-10-16 224444" src="https://github.com/user-attachments/assets/c4126dd7-7613-453f-9056-e8062b867fa9" />

```
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
feature=SelectKBest(mutual_info_classif,k=3)
df1.columns
```
<img width="851" height="41" alt="Screenshot 2025-10-16 224451" src="https://github.com/user-attachments/assets/ee7fc25f-4942-4be7-8072-958c068cc634" />

```
X=df1.iloc[:,0:6]
y=df1.iloc[:,6]
X.columns
```
<img width="749" height="36" alt="Screenshot 2025-10-16 224457" src="https://github.com/user-attachments/assets/f8bd2d43-a36b-402c-b9c2-afcb6eb77963" />

```
feature.fit(X,y)
```
<img width="423" height="181" alt="Screenshot 2025-10-16 224502" src="https://github.com/user-attachments/assets/8f80b349-e872-450d-b80f-81ced1281e9a" />

# RESULT:
Thus, the program to implement Feature Scaling and Feature Selection was completed successfully.
