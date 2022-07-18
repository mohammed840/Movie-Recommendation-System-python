#!/usr/bin/env python
# coding: utf-8

# In[4]:


##Import all necessary libraries
#Get the Movie Titles

import pandas as pd
movies = pd.read_csv("movies.csv")


# In[5]:


movies


# In[6]:


#Get the Movie Titles
import re
def clean_title(title):
   return re.sub("^a-zA-Z0-9]","",title)


# In[7]:


#using pandas apply method to call the function
#takes The title colum and goes through each item in the colum and then pass them to the clean_title function
movies["clean_title"] = movies["title"].apply(clean_title)


# In[8]:


movies


# In[17]:


#using python machine learning libary 
#turning titles into numbers
#using vectorizer to turn the set of titles to a set of number (matrix)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1,2))

tfidf = vectorizer.fit_transform(movies["clean_title"])


# In[15]:


pip install -U scikit-learn scipy matplotlib


# In[90]:


#creating a search engine 
#computing the similarity between a term we enter


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]
    
    return results


# In[91]:


#constructing the interactive search box
#using display function to show diffrent as outputs


import ipywidgets as widgets
from IPython.display import display

movie_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)

#displaying the movie output
#Searches the data set and puts in the output

movie_list = widgets.Output()

def on_type(data):
    with movie_list:
        movie_list.clear_output()
    
        title = data["new"]
        if len(title) > 5:
            display(search(title))

movie_input.observe(on_type, names='value')

#displaying both of the output

display(movie_input, movie_list)


# In[94]:


#reading the ratings.csv file

ratings = pd.read_csv("ratings.csv")


# In[99]:


ratings


# In[102]:


#data type of ratings

ratings.dtypes


# In[108]:


movie_id = 1


# In[111]:


#finding users who likes the same movie
#displaying only unique userid

similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()


# In[112]:


similar_users


# In[119]:


#other movies that the user liked
#any movie that the user rated 4 star


similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]


# In[120]:


similar_user_recs


# In[125]:


# 10 percent of the users that liked the same movie
# value_counts counts how many times each movie appears in the data set

similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

similar_user_recs = similar_user_recs[similar_user_recs > .1]


# In[126]:


similar_user_recs


# In[128]:


#finding user that rated a movie that is in the set of recomdations
all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]


# In[130]:


#finding what percentage all users recommended the movies
all_users_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())


# In[132]:


all_users_recs


# In[133]:


#comparing the percentages using the panda concat method to combine them together 


rec_percentages = pd.concat([similar_user_recs, all_users_recs], axis=1)
rec_percentages.columns = ["similar","all"]


# In[134]:


rec_percentages


# In[135]:


#creating a score which divedes by each other

rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]


# In[136]:


#sorting the recommendations using pandas sort value method
rec_percentages = rec_percentages.sort_values("score", ascending=False)


# In[137]:


rec_percentages


# In[139]:


#acquiring the top 10 reccomdations and then merge it with the movie data 


rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")


# In[140]:


#main reccomendation function
#finding users similar to the movie entered 
#finding all of the users and their recommendations
#creating the score and sorting it
#returns the merged score

def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    similar_user_recs = similar_user_recs[similar_user_recs > .10]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]


# In[149]:


#interactive reccommendation widget


#the input widget
movie_name_input = widgets.Text(
    value="Toy story",
    description="Movie Title:",
    disabled = False

)
#the output widget
recommendation_list = widgets.Output()
#The on_type function
def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            results = search(title)
            movie_id = results.iloc[0]["movieId"]
            display(find_similar_movies(movie_id))

movie_name_input.observe(on_type, names='value')

display(movie_name_input, recommendation_list)


# In[ ]:




