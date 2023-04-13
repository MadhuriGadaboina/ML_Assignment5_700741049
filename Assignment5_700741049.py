#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# read dataset
df = pd.read_csv('CC GENERAL.csv')
# drop CUST_ID column
df.drop('CUST_ID', axis=1, inplace=True)
# drop rows with missing values
df.dropna(inplace=True)

# split dataset into train and test
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

# scale fit training data
scaler = StandardScaler()
scaler.fit(X_train)

# apply transform to training and test data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Apply k-means algorithm on the original data
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_train)
sil_original = silhouette_score(X_train, y_pred)
print('Silhouette score for k-means on original data: ', sil_original)

# apply PCA to training and test data
pca = PCA(n_components=2)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_train)
sil_pca = silhouette_score(X_train, y_pred)
print('Silhouette score for k-means on PCA result: ', sil_pca)


print('Silhouette score for k-means on original data is ', sil_original, ' and silhouette score for k-means on PCA result is ', sil_pca)
if(sil_pca > sil_original):
    print('Silhouette score has improved')
else:
    print('Silhouette score has not improved')
    
# report performance on test data
y_pred = kmeans.predict(X_test)
sil_test = silhouette_score(X_test, y_pred)
print('Silhouette score for k-means on test data: ', sil_test)


# In[9]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Use pd_speech_features.csv
df = pd.read_csv('pd_speech_features.csv')
# drop id column
df.drop('id', axis=1, inplace=True)
# drop rows with missing values
df.dropna(inplace=True)

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# a. Perform Scaling
scaler = StandardScaler()
scaler.fit(X_train)

# apply transform to training and test data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# b. Apply PCA (k=3)
pca = PCA(n_components=3)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# c. Use SVM to report performance
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print('Accuracy score: ', accuracy_score(y_test, y_pred))
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Classification report: ', classification_report(y_test, y_pred))



# In[5]:


import pandas as pd
import numpy as np
# import lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# read dataset
df = pd.read_csv('Iris.csv')
# drop id column
df.drop('Id', axis=1, inplace=True)
# drop rows with missing values
df.dropna(inplace=True)

# split dataset into train and test
X = df.drop('Species', axis=1)
y = df['Species']

# apply LDA to training and test data
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X, y)
X = lda.transform(X)

print(X)


# In[ ]:




