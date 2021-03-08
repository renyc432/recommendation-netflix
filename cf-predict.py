
import os
import pandas as pd
#import numpy as np

from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import cross_validate

#from sklearn.model_selection import train_test_split

#import random

from datetime import datetime
import seaborn as sns

path = "C:\\Users\\roy79\\Desktop\\Research\\recommendation-netflix\\code\\recommendation-netflix"
os.chdir(path)

data = pd.read_csv('..\\..\\data\\data.csv', names=['movie_id','user_id','rating','date'])
movie = pd.read_csv('..\\..\\data\\movie_titles.csv',names=['id','year','title'])

############################## Exploratory Analysis ############################




def predict(svd, user):
    rating_pred = [svd.predict(user, mid).est for mid in movie.id]
    pred_user = pd.concat([movie,pd.Series(rating_pred,name='rating')], axis=1).sort_values('rating', ascending=False)
    return pred_user

  
# trim down dataset so the data will fit in the memory and it results in faster fitting
def process():
    # construct pivot table using only 10M data to see how much data is actually missing
    data_pivot = data[1:10000001].pivot(index='user_id',columns='movie_id',values='rating')
    ct_na = data_pivot.isna().sum().sum()
    pct_na = ct_na/(ct_na+10000000)
    print(f'% rating filled: {(1-pct_na)*100: .2f}%')
        
    review_ct_by_movie = data.groupby('movie_id').size().reset_index(name='count')
    review_ct_by_user = data.groupby('user_id').size().reset_index(name='count')
    
    sns.kdeplot(data=review_ct_by_movie,x='count')
    sns.kdeplot(data=review_ct_by_user,x='count')
    
    review_ct_by_movie['count'].mean()
    review_ct_by_movie['count'].std()
    review_ct_by_user['count'].mean()
    review_ct_by_user['count'].std()

    

def train_ensemble(data, num_batch=4):
    
    reader = Reader()
    batch_size = int(len(data)/num_batch)
    
    batch_index = [(batch_size*(i),batch_size*(i+1)) 
                   if i < (num_batch-1) else (batch_size*(i),len(data)) 
                   for i in range(num_batch)]
    svds = {}
    for batch in batch_index:
        start = batch[0]
        end = batch[1]
        svd = SVD()
        data_sp = Dataset.load_from_df(data[['user_id','movie_id','rating']][start:end], reader)
        train = data_sp.build_full_trainset()
        
        # The full dataset took 1:18:20 to fit
        start = datetime.now()
        svd.fit(train)
        print(datetime.now()-start)
        svds[(start,end)] = svd
    return svds    
    

def train(data):
    reader = Reader()
    svd = SVD()

    data_sp = Dataset.load_from_df(data[['user_id','movie_id','rating']], reader)
    
    # this will take too long to run, cv with a smaller dataset
    svd_cv = cross_validate(svd, data_sp, measures=['rmse'],cv=5,verbose=True)

    # The full dataset is too big to be processed at once, so we divide the dataset into batches
    train = data_sp.build_full_trainset()
    # The full dataset took 1:18:20 to fit
    svd.fit(train)
    return svd
    
    
def main():
    
    
    svd = train(data)
    
    pred = [(i,predict(svd,i)) for i in range(1,1000000, 43123)]
    
    [(print(user[0]),print(user[1].head(10)), print(user[1].tail(10)), print('\n')) for user in pred]




if __name__ == "__main__":
    main()

