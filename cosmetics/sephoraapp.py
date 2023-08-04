import pandas as pd
import streamlit as st
import numpy as np
import re
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 
import tkinter
import matplotlib.backends.backend_tkagg





df= pd.read_excel("/Users/madisonritchie/Library/CloudStorage/OneDrive-SharedLibraries-Onedrive/Documents/Data Science Classes/Research/cosmetics.xlsx")
st.title("Visualizing the Sephora Data Set")
st.subheader("Original Data")
st.dataframe(df)
st.subheader("Transformed Data")
k_df = df

#Create Regression Dataset
df = df.assign(Ingredients=df.Ingredients.str.split(',').explode('Ingredients'))
df['Ingredients']= [re.sub(r'[\)\(]', '', str(x)) for x in df['Ingredients']]
st.dataframe(df)

#Perform Linear Regression
lin_reg = smf.ols("Rank ~ Ingredients", data=df).fit()
st.write(lin_reg.summary())

#Visualize Linear Regression


#Perform K-Clustering
k_df = k_df.drop(columns= ['Ingredients', 'Brand'])
k_df = pd.get_dummies(k_df, columns=['Label'] )
k_df = k_df.set_index('Name')
st.dataframe(k_df)
scaled_df = StandardScaler().fit_transform(k_df)
st.dataframe(scaled_df)
#instantiate the k-means class, using optimal number of clusters
kmeans = KMeans(init="random", n_clusters=20, n_init=10, random_state=0)
km = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
#fit k-means algorithm to data
kmeans.fit(scaled_df)
label = kmeans.fit(scaled_df)
#view cluster assignments for each observation
kmeans.labels_
k_df['cluster'] = kmeans.labels_
st.dataframe(k_df)
silhouette_score_average = silhouette_score(scaled_df, kmeans.predict(scaled_df))
st.dataframe(k_df)
st.subheader("silhouette_score_average")
st.write(silhouette_score_average)

#Create Kmeans Scatterplot
plt.scatter(scaled_df[:,0], 
            scaled_df[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1], 
            s=200,                            
            c='red')     
plt.show()