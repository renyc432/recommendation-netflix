
import os
import pandas as pd
import numpy as np

from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import cross_validate

from sklearn.model_selection import train_test_split

import random

path = "C:\\Users\\roy79\\Desktop\\Research\\recommendation-netflix"
os.chdir(path)

movie = pd.read_csv('movie_titles.csv',names=['id','year','title'])



def predict(svd, user):
    rating_pred = [svd.predict(user, mid).est for mid in movie.id]
    pred_user = pd.concat([movie,pd.Series(rating_pred,name='rating')], axis=1).sort_values('rating', ascending=False)
    return pred_user

    
    
def main():
    data = pd.read_csv('data.csv', names=['movie_id','user_id','rating','date'])
    
    reader = Reader()
    
    
    num_batch = 10
    batch_size = int(len(data)/10)
    
    batch_index = [(batch_size*(i),batch_size*(i+1)) if i < 9 else (batch_size*(i),len(data)) for i in range(10)]
    
    svd = SVD()
    #svd_cv = cross_validate(svd, data_sp, measures=['rmse'],cv=5,verbose=True)

    for batch in batch_index:
        start = batch[0]
        end = batch[1]
        data_sp = Dataset.load_from_df(data[['user_id','movie_id','rating']][start:end], reader)
        train = data_sp.build_full_trainset()
        svd.fit(train)
        

    
    pred = [(i,predict(svd,i)) for i in range(1,1000000, 43123)]
    util = [(print(user[0]),print(user[1].head(10)), print(user[1].tail(10)), print('\n')) for user in pred]




if __name__ == "__main__":
    main()

