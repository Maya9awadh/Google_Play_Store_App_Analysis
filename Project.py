# -*- coding: utf-8 -*-
"""
Maya Al-hatmi
124062
"""
#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#read the data 
data=pd.read_csv("googleplaystore.csv")

#data cleaning

#delete rows that create issue in modelling
data=data.drop([10472])

#drop nan value
data=data.dropna(how='any',axis=0)

#drop dublicate row
data.drop_duplicates(subset=['App'],keep=False)

#drop unusefel charectar from variable size
data["Size"] = data["Size"].apply(lambda x: str(x).replace(",", "") if "," in str(x) else x)
data["Size"] = data["Size"].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
data["Size"] = data["Size"].apply(lambda x: str(x).replace("Varies with device", "NAN") if "Varies with device" in str(x) else x)
data["Size"] = data["Size"].apply(lambda x: float(str(x).replace('k', '')) / 1024 if 'k' in str(x) else x)
data["Size"] = data["Size"].apply(lambda x:x.replace("+","")if "+" in str(x) else x)


#convert size to float and replace 'NAN' in size column with mean of the column
data["Size"] = data["Size"].apply(lambda x:float(x))
data["Size"].fillna((data["Size"].mean()), inplace=True)

#Q1) what category of apps is most prevalent among tenager
data1=data.loc[data['Content Rating']=='Teen']
#draw pie chart of category and content
labels = data1['Category'].value_counts().index.tolist()
sizes = [round(item,3) for item in list(data1['Category'].value_counts()/data1.shape[0])]      
fig1, ax1 = plt.subplots(figsize = (15,15))
ax1.pie(sizes , labels=labels, rotatelabels=True,autopct='%1.1f%%',
shadow=True, startangle=90)
ax1.axis('equal')#ensures that pie is drawn as a circle.
plt.title("Category Distribution",size = 20,loc = "left")
plt.show()


#Q2)How bussiness apps compared to lifestyle apps according to 
#draw bar chart to represent the size of each gategory
Size=data["Size"]
Category=data['Category']
fig=plt.figure(figsize=(100,100))
ax=fig.add_subplot(111)
rect1=ax.bar(Category,Size,align='center')
plt.xticks(rotation=90,ha='right',fontsize=100)
plt.show()

#Q3)What is the relationship between size of app 
#(as independent variable) and rating of app as outcome variable.

#build regression model
x=data['Size']
y=data['Rating']
plt.figure(figsize=(10,8))
plt.scatter(x,y)
theta1,theata0=np.polyfit(x,y,1)
plt.plot(x,theta1*x + theata0,color='black')
plt.ylabel('Rating of app')
plt.xlabel('Size of app')
plt.title('Relationship between Rating and size')
plt.show()
print("weight of X in regression model",theta1)
print("bias term = ", theata0)
print('correlation coefficient: ',np.corrcoef(x,y))


#load data from the seconed file 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
df1 = pd.read_csv('googleplaystore_user_reviews.csv')
data2=df1.dropna(how='any',axis=0)
x=data2["Translated_Review"]
y=data2["Sentiment"]
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
X_train_raw, X_test_raw, y_train, y_test =train_test_split(x, y, test_size=0.2,shuffle=False)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
X_test = vectorizer.transform( X_test_raw  )
predictions = classifier.predict(X_test)
print(predictions)

#calculate the accuracy
#calculate the accuracy
score=classifier.score(X_test ,y_test)
print("The accuracy of logistic regression: ",score)

#clustring
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Standardize the data
df =  df1.iloc[:, 3:5]
#dropping the rows which has null value 
final = df.dropna(how='any',axis=0)
X_std = StandardScaler().fit_transform(final)
# Run local implementation of kmeans
km = KMeans()
km = KMeans(n_clusters=2, max_iter=1000)
km.fit(X_std)
centroids = km.cluster_centers_
# Plot the clustered data
fig, ax = plt.subplots(figsize=(6, 6))
plt.scatter(X_std[km.labels_ == 0, 0], X_std[km.labels_ == 0, 1],
            c='green', label ='cluster 1')
plt.scatter(X_std[km.labels_ == 1, 0], X_std[km.labels_ == 1, 1],
            c='blue', label ='cluster 2')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300,
            c='r', label='centroid')
plt.legend()
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('Sentiment_Polarity')
plt.ylabel('Sentiment_Subjectivity')
plt.title('Visualization of clustered data', fontweight='bold')
ax.set_aspect('equal')
plt.show()