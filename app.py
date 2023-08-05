import pandas as pd
import streamlit as st
import numpy as np
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
from sklearn.metrics import silhouette_score 
from PIL import Image


#read in data
df= pd.read_excel('cosmetics.xlsx')

#App Interface
st.title("💄 Visualizing the Sephora Dataset")
st.write("This app pairs with a research paper analyzing recent trends in specialty beauty cosmetics.  Download the PDF of the research paper in the sidebar.")
st.write("Navigate the app by opening the sidebar (arrow on left corner) and interacting with widgets.")
st.write("The dataset includes product name, brand, type, price, review rating (rank), target skin type, and ingredients lists for 1472 cosmetics found from Sephora via word embedding.")
st.markdown("Visit the original dataset at: https://www.kaggle.com/datasets/kingabzpro/cosmetics-datasets")
image = Image.open('sephora.jpeg')
st.sidebar.image(image, width= 200)
st.sidebar.title("Navigation")
navigation = st.sidebar.radio("navigation", label_visibility="hidden", options= ('🧼 Original Dataset', '💄 Linear Regression', '🧴 K-Means Clustering'))
st.sidebar.button("Download Research Paper")
k_df = df
df = df.assign(Ingredients=df.Ingredients.str.split(',').explode('Ingredients'))
df['Ingredients']= [re.sub(r'[\)\(]', '', str(x)) for x in df['Ingredients']]

#Display original dataset
if navigation == "🧼 Original Dataset":
    st.subheader("Original Dataset")
    st.dataframe(df)
    st.markdown("*:red[Figure 1]*")
    st.subheader("Ingredient Breakdown")
    ingredientlist = df["Ingredients"].to_list()
    select = st.multiselect("", options=ingredientlist)
    filtereddf= df
    filtereddf= filtereddf[filtereddf.Ingredients.isin(select)]
    st.dataframe(filtereddf)
    st.markdown("*:red[Figure 2]*")
    

#Perform Linear Regression
if navigation == '💄 Linear Regression':
    st.subheader("Linear Regression Test")
    st.write("A linear regression test is a mathematical test used for evaluating and quantifying the relationship between the considered variables.")
    lin_reg = smf.ols("Rank ~ Ingredients", data=df).fit()
    st.write("A linear regression test (Ordinary-Least Squares) was performed with product ingredient as the independent variable and rating as the dependent variables utilizing the Statsmodel package.")
    details = st.button("Press for Linear Regression Summary")
    st.write("The R-squared value was determined to be 0.567, the F-statistic was 1.179, and the corresponding p-value was 0.0131.  ")
    if details:
        st.write(lin_reg.summary())
        st.markdown("*:red[Figure 3]*")
    #Visualize Linear Regression
    X = df['Ingredients'].values
    y = df['Rank'].values
    linregscatter, ax = plt.subplots(figsize=(10,15))
    plt.scatter(X, y,color='g')
    plt.title("Ingredients vs Rating Scatterplot")
    plt.xlabel("Ingredient")
    plt.ylabel("Rating")
    plt.xticks(rotation=90)
    st.pyplot(linregscatter)
    st.markdown("*:red[Figure 4]*")

#Perform K-Clustering
if navigation == '🧴 K-Means Clustering':
    st.title("K-Means Clustering")
    st.write("A k-clustering test works by finding patterns in the data set and grouping those with similarities together and those with differences in different clusters.")
    k_df = k_df.drop(columns= ['Ingredients', 'Brand'])
    k_df = pd.get_dummies(k_df, columns=['Label'] )
    k_df = k_df.set_index('Name')
    initialdf = st.button("Expand for Initial Dataset")
    if initialdf:
        st.subheader("Initial K-Means Dataset")
        st.dataframe(k_df)
        st.markdown("*:red[Figure 5]*")
    scaledbutton = st.button("Expand for Scaled Dataset")
    scaled_df = StandardScaler().fit_transform(k_df)
    if scaledbutton:
        st.subheader("Scaled Dataset")
        st.write("Data was scaled beforehand due to the mixed numerical nature of the data ")
        st.dataframe(scaled_df)
        st.markdown("*:red[Figure 6]*")
    #instantiate the k-means class, using optimal number of clusters
    kmeans = KMeans(init="random", n_clusters=20, n_init=10, random_state=0)
    km = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
    #fit k-means algorithm to data
    kmeans.fit(scaled_df)
    label = kmeans.fit(scaled_df)
    #view cluster assignments for each observation
    k_df['Cluster'] = kmeans.labels_
    #calculate the silhouette score
    silhouette_score_average = silhouette_score(scaled_df, kmeans.predict(scaled_df))
    st.title("K_Means Clustering Results")
    st.write("Each product is assigned to the most similar cluster in the Cluster column.  It was determined that 20 clusters was ideal through the Elbow method.")
    st.write("For the k-clustering test individual products will be grouped by the numerical input variables price and rank in addition to the binary variables target skin type and product category.")
    st.dataframe(k_df)
    st.markdown("*:red[Figure 7]*")
    #Create Kmeans Scatterplot
    kmeanscatter, ax = plt.subplots()
    plt.scatter(scaled_df[:,0], scaled_df[:,1])
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red')   
    plt.title("Scaled Dataset Scatterplot and Clusters")  
    st.title("Scaled Dataset Scatterplot and Clusters")
    st.pyplot(kmeanscatter)
    st.markdown("*:red[Figure 8]*")
    st.write("Blue points represent individual products while red points represent the 20 clusters.")
    st.title("Silhouette Score")
    st.write("A silhouette score varies in width between -1 and 1 with a negative value relating to greater dissimilarity.  The greater a positive value the more likely the higher likelihood the element is clustered in the correct group.")
    st.write("While researchers are not in unanimous agreement, in general a silhouette index below 0.4 indicates weak clusters.")
    st.subheader("silhouette_score_average")
    st.write(silhouette_score_average)
    st.markdown("*:red[Figure 9]*")
