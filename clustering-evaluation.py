import os
import pandas as pd
import re
import csv
import time
path = "C:\\Users\\roy79\\Desktop\\Research\\recommendation-netflix\\code\\recommendation-netflix"
os.chdir(path)

imdb = pd.read_csv('..\\..\\data\\IMDb-movies.csv')
netflix_id = pd.read_csv('..\\..\\data\\netflix_id.csv',names=['id','year','title'])

############################# Clean Data
imdb.drop_duplicates('title', inplace=True)
netflix_imdb = pd.merge(netflix_id, imdb, how='inner', left_on=['year','title'], right_on=['year','title'])


def clean_title(title, regex):
    clean_title = re.sub(regex, ' ', title.lower()).strip()
    return clean_title

regex = '[^a-zA-Z0-9]+'

imdb_clean_title = [clean_title(title, regex) for title in imdb['title']]
netflix_clean_title = [clean_title(title, regex) for title in netflix_id['title']]

imdb['clean_title'] = imdb_clean_title
netflix_id['clean_title'] = netflix_clean_title

############################# label imdb rows that have a match on netflix
imdb['netflix_movie_id'] = -1
for imdb_id in imdb.index:
    print(imdb_id)
    net_index = netflix_id.index[netflix_id['clean_title'] == imdb['clean_title'][imdb_id]].tolist()
    if net_index == []:
        imdb['netflix_movie_id'][imdb_id] = 0
    else:
        imdb['netflix_movie_id'][imdb_id] = net_index[0]+1


sum(imdb['netflix_movie_id'] != 0)
# 3637

imdb.to_csv('imdb_labeled_netflix.csv')


########################## load csv
imdb_labeled = pd.read_csv('imdb_labeled_netflix.csv')
netflix_id = pd.read_csv('..\\..\\data\\netflix_id.csv',names=['id','year','title'])
rating = pd.read_csv('..\\..\\data\\data.csv', names=['movie_id','user_id','rating','date'])

# sanity check
netflix_id[netflix_id['id'] == 5900]
netflix_id[netflix_id['id'] == 10898]

imdb_netflix = imdb_labeled[imdb_labeled['netflix_movie_id'] != 0]
rating_content_based = rating.loc[rating['movie_id'].isin(imdb_netflix['netflix_movie_id'])]

rating_content_based.drop('date', axis=1, inplace=True)
rating_content_based.to_csv('rating_imdb_subset.csv', index=False)




# =============================================================================
# rating_gby_user = rating_content_based.groupby('user_id')['movie_id'].count()
# sum(rating_gby_user > 100)
# # 85150
# =============================================================================


# find users who have given a 5 to 20 or more movies that's overlapping
good_ratings_content_based = rating_content_based[rating_content_based['rating']==5]
len(good_ratings_content_based)
good_rating_gby_user = good_ratings_content_based.groupby('user_id')['movie_id'].count()
user_5 = good_rating_gby_user[good_rating_gby_user >= 20].index

# get the 5 ratings of these users
rating_user5 = good_ratings_content_based[good_ratings_content_based['user_id'].isin(user_5)]
rating_user5.sort_values('user_id', inplace=True)
























