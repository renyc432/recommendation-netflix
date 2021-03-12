
import pandas as pd
import re
import csv
import time
from datetime import datetime


############################# Load Data

# https://www.kaggle.com/shivamb/netflix-shows
# https://www.kaggle.com/nishanthkv/netflix

netflix = pd.read_csv('..\\..\\data\\netflix_titles.csv')
netflix_2 = pd.read_csv('..\\..\\data\\netflix_titles_2.csv')


imdb = pd.read_csv('..\\..\\data\\IMDb-movies.csv')

netflix_id = pd.read_csv('..\\..\\data\\netflix_id.csv',names=['id','year','title'])


############################# Clean Data
netflix_combined = pd.concat([netflix, netflix_2], axis=0)
netflix_combined.drop_duplicates('title', inplace=True)
imdb.drop_duplicates('title', inplace=True)
netflix_imdb = pd.merge(netflix_id, imdb, how='inner', left_on=['year','title'], right_on=['year','title'])


def clean_title(title, regex):
    clean_title = re.sub(regex, ' ', title.lower()).strip()
    return clean_title
    

regex = '[^a-zA-Z0-9]+'
imdb_clean_title = [clean_title(title, regex) for title in imdb['title']]
netflix_clean_title = [clean_title(title, regex) for title in netflix_id['title']]



#################### METHOD 1: IMDB dataset contains 3635 movies that Netflix Prize Data also has
overlap_title = list(set(imdb_clean_title).intersection(netflix_clean_title))
print(f'overlapping titles: {len(overlap_title)}')
# 3635


#################### METHOD 2: Anither two Netflix datasets contain 660 movies that Netflix Prize Data also has
netflix_combined_clean_title = [clean_title(title, regex) for title in netflix_combined['title']]

overlap_title_netflix = list(set(netflix_combined_clean_title).intersection(netflix_clean_title))
print(len(overlap_title_netflix))
# 660


# =============================================================================
# # Q: does the same tv show over many seasons share the same id?
# # A: no, they follow this format: 'Friends: Season 5'/'The Office: Season 1'/'The Office: Series 2'
# friends = netflix_id[netflix_id['title'].str.contains('Friends')]
# office = netflix_id[netflix_id['title'].str.contains('Office')]
# =============================================================================



#################### METHOD 3: Retrieve IMDb objects using IMDbPy tool: estimatedly this will get us 7000 movies that Netflix Prize Data has

# pip install imdbpy
from imdb import IMDb

ia = IMDb()


title = 'example title'
ia.search_movie(title)

# search_keyword() doesn't work well, too slow

def search_for_movie(netflix_title):
    result = ia.search_movie(netflix_title)
    if result != []:
        top = result[0]
        ia.update(top)
        title = top.get('title')
        year = top.get('year')
        genres = top.get('genres')
        abstract = top.get('plot')
        lang = top.get('language codes')
        country = top.get('country codes')
        return [netflix_title, title, year, genres, abstract, lang, country]
    return None


num_movies = len(netflix_id['title'])
num_batches = 100
batch_size = int(num_movies/num_batches)


imdb_objects = []

num_movies

batch = {0:range(0,2000),
         1:range(2000,4000),
         2:range(4000,6000),
         3:range(6000,8000),
         4:range(8000,10000),
         5:range(10000,12000),
         6:range(12000,14000),
         7:range(14000,16000),
         8:range(16000, num_movies)}


################## TO CHANGE
batch_id = 0
col_names = ['netflix_title','title','year','genres','abstract','lang','country']


mybatch = batch[batch_id]

output = open('.\\data\\IMDB-' + batch_id + time.strftime("%Y%m%d-%H%M%S") + '.csv','w',encoding='utf-8',newline='')
writer = csv.writer(output)
writer.writerow(col_names)

# if failed to fetch, append here
missed_titles = []

for title in netflix_id['title'][mybatch]:
    try:
        imdb_object = search_for_movie(title)
        writer.writerow(imdb_object)
    except:
        missed_titles.append(title)
        time.sleep(1800)
        continue

pd.Series(missed_titles).to_csv(f'.\\data\\missed_titles-{batch_id}.csv')



# sum(netflix_object_df['year'] == netflix_id['year'])









