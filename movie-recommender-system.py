#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import pandas as pd


# In[2]:


#load the datasets
movies_df = pd.read_csv('tmdb_5000_movies.csv')
credits_df = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


#printing first 3 records of movies dataset
movies_df.head(3)


# In[4]:


#printing first 3 records of credits dataset
credits_df.head(3)


# In[5]:


#printing first value of cast column in credits dataset
credits_df.head(1)['cast'].values


# In[6]:


#printing first value of crew column in credits dataset
credits_df.head(1)['crew'].values


# In[7]:


#printing shape of movies_df
movies_df.shape


# In[8]:


#printing shape of credits_df
credits_df.shape


# In[9]:


#merging both datasets into movies_df on basis of 'title' column
movies_df = movies_df.merge(credits_df,on='title')


# In[10]:


movies_df.shape


# In[11]:


movies_df.head(1)


# In[12]:


movies_df.info()


# In[13]:


#------Dropping Columns------
# budget
# homepage
# id
# original_language
# original_title
# popularity
# production_comapny
# production_countries
# release-date(not sure)
# revenue
# runtime
# spoken_languages 
# status
# tagline
# vote_average
# vote_count
movies_df = movies_df.drop(['budget','homepage','id','original_language','original_title','popularity','production_companies','production_countries'
             ,'release_date','revenue','runtime','spoken_languages','status','tagline','vote_average','vote_count'],axis = 1)


# In[14]:


movies_df.head(3)


# In[15]:


#checking missing values
movies_df.isnull().sum()


# In[16]:


#dropping null rows as we have only 3 rows that are null
movies_df.dropna(inplace=True)


# In[17]:


movies_df.isnull().sum()


# In[18]:


#checking whether we have duplicated data or not
movies_df.duplicated().sum()


# In[19]:


movies_df.iloc[0].genres


# In[20]:


#function to merge all genre names of particular movie
# from=> '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
# to=> [Action, Adventure, Fantasy, Science Fiction]

#we all also convert string into list with help of ast
#eg. from=> '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
# to=> [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]
import ast
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 


# In[21]:


movies_df['genres'] = movies_df['genres'].apply(convert)
movies_df.head()


# In[22]:


movies_df['keywords'] = movies_df['keywords'].apply(convert)
movies_df.head()


# In[23]:


#displaying 1st row of cast column
movies_df['cast'][0]


# In[24]:


#here we are creating function that will take first 3 dictionery from each record of cast column and extract the names of character
# eg. FROM {"cast_id": 242, "character": "Jake Sully", "credit_id": "5602a8a7c3a3685532001c9a", "gender": 2, "id": 65731, "name": "Sam Worthington", "order": 0},
# {"cast_id": 3, "character": "Neytiri", "credit_id": "52fe48009251416c750ac9cb", "gender": 1, "id": 8691, "name": "Zoe Saldana", "order": 1},
# {"cast_id": 25, "character": "Dr. Grace Augustine", "credit_id": "52fe48009251416c750aca39", "gender": 1, "id": 10205, "name": "Sigourney Weaver", "order": 2}

# TO [Sam Worthington, Zoe Saldana, Sigourney Weaver]
import ast
def top_cast_names(text):
    L = []
    counter = 0 
    for i in ast.literal_eval(text):
        if counter!=3:
            L.append(i['name']) 
            counter+=1
        else:
            break
    return L 


# In[25]:


movies_df['cast'] = movies_df['cast'].apply(top_cast_names)


# In[26]:


movies_df.head()


# In[27]:


import ast
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job']=='Director':
            L.append(i['name']) 
            break
    return L 


# In[28]:


movies_df['crew'] = movies_df['crew'].apply(fetch_director)


# In[29]:


movies_df.head()


# In[30]:


#converting movies 'overview' from string to list
movies_df['overview'] = movies_df['overview'].apply(lambda x:x.split())


# In[31]:


movies_df.head()


# In[32]:


#now we will apply tranformation on all columns as we need to remove space between 2 words
# eg. FROM Sam Worthington TO SamWorthington
# we are doing this because in further notebook we are going to create TAGS, when we create TAGS two different TAGS will be 
# created for 'Sam Worthington' words. As we know it is single entity so we need to merge this kind of words and then TAG will be 
# like "SamWorthington" instead of "Sam" "Worthington"

#we will apply this to all columns

movies_df['genres'] = movies_df['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['cast'] = movies_df['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['crew'] = movies_df['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[33]:


movies_df.head()


# In[34]:


#concatenating Overview, Genres, Keywords, Cast and crew into Tags column
movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast'] + movies_df['crew']


# In[35]:


movies_df.head()


# In[36]:


#creating new dataframe with 3 important columns
new_movies_df = movies_df[['movie_id','title','tags']]


# In[37]:


new_movies_df.head()


# In[38]:


#converting list to string in tags columns
new_movies_df['tags'] = new_movies_df['tags'].apply(lambda x:" ".join(x))


# In[39]:


new_movies_df['tags'][0]


# In[40]:


#converting text of 'Tags' column into lowercase
new_movies_df['tags'] = new_movies_df['tags'].apply(lambda x:x.lower())


# In[41]:


new_movies_df.head()


# In[42]:


#installing nltk library
get_ipython().system('pip install nltk')


# In[43]:


#stemming
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def text_stem(text):
    l = []
    for i in text.split():
        l.append(ps.stem(i))
        
    return " ".join(l)


# In[44]:


new_movies_df['tags'] = new_movies_df['tags'].apply(text_stem)


# In[45]:


#text representative
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words = 'english')


# In[46]:


vectors = cv.fit_transform(new_movies_df['tags']).toarray()


# In[47]:


#shape of vectors with 4806 movies and 5000 words
vectors.shape


# In[48]:


#displaying 5000 features
cv.get_feature_names()


# In[50]:


#here we will calculate cosine_similarities between each vectors 
from sklearn.metrics.pairwise import cosine_similarity
similarity_result = cosine_similarity(vectors)


# In[51]:


#shape of similarity
similarity_result.shape


# In[55]:


#sorted(list(enumerate(similarity_result[0])), reverse = True, key = lambda x:x[1])[1:6]


# In[61]:


def recommendation(movie):
    movie_index = new_movies_df[new_movies_df['title'] == movie].index[0]
    distances = similarity_result[movie_index]
    movie_list =sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_movies_df.iloc[i[0]].title)
        #print(i[0])


# In[62]:


#recommendation('Avatar')
#new_movies_df.iloc[1216]

recommendation('Avatar')


# In[63]:


recommendation('Batman Begins')


# In[67]:


import pickle
pickle.dump(new_movies_df.to_dict(),open('movies.pkl','wb'))


# In[68]:


pickle.dump(similarity_result,open('similarity.pkl','wb'))


# In[ ]:




