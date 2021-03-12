import os
import pandas as pd
import numpy as np

from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import cross_validate

from sklearn.metrics import mean_squared_error
#import random

from surprise.model_selection import RandomizedSearchCV

from datetime import datetime
import seaborn as sns

from joblib import dump, load

path = "C:\\Users\\roy79\\Desktop\\Research\\recommendation-netflix\\code\\recommendation-netflix"
os.chdir(path)

data = pd.read_csv('..\\..\\data\\data.csv', names=['movie_id','user_id','rating','date'])
netflix_id = pd.read_csv('..\\..\\data\\netflix_id.csv',names=['id','year','title'])




# Todo 1: add svd hyperparameter tuning
# Todo 2: svd cross validation on a smaller dataset, print result


class SVD_Ensemble:

    def fit(self, data, num_batch=10):
        self.num_batch=num_batch
        self.movie_ids = data.movie_id.unique()
        self.user_ids = data.user_id.unique()
        self.user_ids.sort()
        
        num_user = len(self.user_ids)
        
        batch_size = int(num_user / num_batch)
        self.batch_user_index = {}
        for i in range(num_batch):
            if i == num_batch-1:
                self.batch_user_index[i] = self.user_ids[batch_size*(i):num_user]
            else:
                self.batch_user_index[i] = self.user_ids[batch_size*(i):batch_size*(i+1)]
        
        train = data[['user_id','movie_id','rating']]
        
        svds = {}
        reader = Reader()
        for i in range(num_batch):
            batch = self.batch_user_index[i]

            is_batch = train['user_id'].isin(batch)
            train_batch = train[is_batch]
            
            svd = SVD()
            data_sp = Dataset.load_from_df(train_batch, reader)
            train_batch_loaded = data_sp.build_full_trainset()
            
            svd.fit(train_batch_loaded)
            svds[i] = svd
        self.svds = svds  
    
    
    # if you just want to use the svd trained on the subset of users that contain the target user, then use weight=1
    # else take weighted average of all svds
    def predict(self, user, weight=0.8):
        
        assert(weight <= 1)
        
        # find the svd model
        contains_user = [user in self.batch_user_index[i] for i in range(self.num_batch)]
        new_user = False
        if True not in contains_user:
            new_user = True
            weight = 1/self.num_batch
        else:
            user_batch_id = contains_user.index(True)
            
        
        # if a new user, ensemble svds by average
        if new_user:
            ratings = []
            for i in range(self.num_batch):
                rating = [self.svds[i].predict(user, mid).est for mid in self.movie_ids]
                ratings.append(rating)
            rating_avg = np.average(ratings, axis=0)
            output = pd.Series(rating_avg, index=self.movie_ids).sort_values(ascending=False)
            return output

        
        # if weight=1, then use a single svd trained on the user_id
        if weight == 1:
            rating = [self.svds[user_batch_id].predict(user, mid).est for mid in self.movie_ids]
            output = pd.Series(rating, index=self.movie_ids).sort_values(ascending=False)
            return output
        
        
        # if weight < 1, then use an ensemble of svd
        user_weight = weight
        else_weight = (1-weight)/(self.num_batch-1)
        
        ratings = []
        for i in range(self.num_batch):
            rating = [self.svds[i].predict(user, mid).est for mid in self.movie_ids]
            if i == user_batch_id:
                rating = np.multiply(rating, user_weight)
            else:
                rating = np.multiply(rating, else_weight)
            ratings.append(rating)
        
        rating_weighted_avg = np.sum(ratings, axis=0)
        output = pd.Series(rating_weighted_avg, index=self.movie_ids).sort_values(ascending=False)
        return output
    
    def evaluate(self):
        pass


def train(data):
    reader = Reader()
    svd = SVD()

    data_sp = Dataset.load_from_df(data[['user_id','movie_id','rating']], reader)

    # The full dataset is too big to be processed at once, so we divide the dataset into batches
    train = data_sp.build_full_trainset()
    # The full dataset took 1:18:20 to fit
    svd.fit(train)
    return svd
    
def predict(svd, user):
    rating_pred = [svd.predict(user, mid).est for mid in netflix_id.id]
    pred_user = pd.concat([netflix_id,pd.Series(rating_pred,name='rating')], axis=1).sort_values('rating', ascending=False)
    return pred_user
    

def exploratory(data):
    # construct pivot table using only 10M data to see how much data is actually missing
    data_pivot = data[1:10000001].pivot(index='user_id',columns='movie_id',values='rating')
    ct_na = data_pivot.isna().sum().sum()
    pct_na = ct_na/(ct_na+10000000)
    print(f'% rating filled: {(1-pct_na)*100: .2f}%')
    
    col_ct = 'Rating Count'
    review_ct_by_movie = data.groupby('movie_id').size().reset_index(name=col_ct)
    review_ct_by_user = data.groupby('user_id').size().reset_index(name=col_ct)
    
    sns.kdeplot(data=review_ct_by_movie,x=col_ct).set_title('Rating Density by Movie')
    sns.kdeplot(data=review_ct_by_user,x=col_ct).set_title('Rating Density by User')
    
    print('Rating count by movie - mean:', review_ct_by_movie[col_ct].mean())
    print('Rating count by movie - sd:', review_ct_by_movie[col_ct].std())
    print('Rating count by user - mean:', review_ct_by_user[col_ct].mean())
    print('Rating count by user - sd:', review_ct_by_user[col_ct].std())
    
    return data


def hyperparameter_tuning(data):
    reader = Reader()
    data_rs = Dataset.load_from_df(data, reader)
    
    parameters = {'n_factors':[50,100,200],'n_epochs':[20,40],'lr_all':[0.005,0.001], 'reg_all':[0.05,0.02,0.01]}
    
    rs = RandomizedSearchCV(SVD, parameters, measures=['rmse'], cv=5)
    rs.fit(data_rs)
    
    return rs.best_params['rmse'], rs.best_score
    
    

def main():
    
    exploratory(data)
    
    start = datetime.now()
    params = hyperparameter_tuning(data[['user_id','movie_id','rating']][0:1000000])
    print(datetime.now()-start)
    
    reader = Reader()
    svd = SVD()
    
    data_sp_cv = Dataset.load_from_df(data[['user_id','movie_id','rating']][0:1000000], reader)
    svd_cv = cross_validate(svd, data_sp_cv, measures=['rmse'],cv=5,verbose=True)
    
    
    method = 'one-off'
    if method == 'one-off':
        svd = train(data)
        dump(svd, 'cf-svd.joblib')
    elif method == 'ensemble':
        trainer = SVD_Ensemble()
        trainer.fit(data)
        for i in range(len(trainer.svds)):
            dump(trainer.svds[i], f'cf-svd-ensemble-{i}.joblib')
    
#    pred = [(i,predict(svd,i)) for i in range(1,1000000, 43123)]
    
#    [(print(user[0]),print(user[1].head(10)), print(user[1].tail(10)), print('\n')) for user in pred]
    
    
    trainer.predict(1789485)
    


if __name__ == "__main__":
    main()

