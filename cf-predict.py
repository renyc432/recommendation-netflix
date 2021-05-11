import os
import pandas as pd
import numpy as np

from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error

from datetime import datetime
import seaborn as sns

import random

from joblib import dump, load

path = "C:\\Users\\roy79\\Desktop\\Research\\recommendation-netflix\\code\\recommendation-netflix"
os.chdir(path)

data = pd.read_csv('..\\..\\data\\data.csv', names=['movie_id','user_id','rating','date'])
netflix_id = pd.read_csv('..\\..\\data\\netflix_id.csv',names=['id','year','title'])


data[0:100000].to_csv('Netflix Prize Data - Sample.csv')

################################### Summary ####################################

# time
# hyperparameter tuning (1M): ~01:00:00
# 5 fold CV (10M): ~00:20:00
# 5 fold CV (100M):

# fit - onestop: ~01:20:00
# fit - ensemble: ~01:20:00

# result
# hyperparameters: n_factors = 50, n_epochs = 20, lr_all = 0.005, reg_all = 0.05
# 5 fold CV (10M) rmse: 0.9384

################################ Ensembled SVD #################################
class SVD_Ensemble:

    def fit(self, data, 
            num_batch=10, 
            n_factors=50,
            n_epochs=20,
            lr_all=0.005,
            reg_all=0.05):
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
            
            svd = SVD(n_factors=n_factors,
                      n_epochs=n_epochs,
                      lr_all=lr_all,
                      reg_all=reg_all)
            data_sp = Dataset.load_from_df(train_batch, reader)
            train_batch_loaded = data_sp.build_full_trainset()
            
            svd.fit(train_batch_loaded)
            svds[i] = svd
        self.svds = svds  
    
    
    # if you just want to use the svd trained on the subset of users that contain the target user, then use weight=1
    # else take weighted average of all svds
    def predict_user(self, user, weight=0.8):
        
        assert(weight <= 1)
        
        # find the svd model trained on the user
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
        
        # if weight=1, then use a single svd trained on the user_id
        elif weight == 1:
            rating_avg = [self.svds[user_batch_id].predict(user, mid).est for mid in self.movie_ids]
        else:
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
            rating_avg = list(np.sum(ratings, axis=0))

        output = pd.DataFrame({'rating_pred':rating_avg, 
                               'movie_id':list(self.movie_ids), 
                               'user_id':[user]*len(self.movie_ids)})
        return output
    
    def predict(self, users, weight=0.8):
        preds = []
        for user in users:
            pred = self.predict_user(user, weight=weight)
            preds.append(pred)
        return pd.concat(preds,axis=0)
            
            
    def evaluate(self, data, test_size = 0.2, weight=0.8):
        testset = data.sample(frac=test_size)
        trainset = data.drop(testset.index)
        print('retrieve svds')
        try:
            self.svds
        except:
            print(1)
            self.fit(trainset)
        print('predict')
        y_pred = self.predict(testset.user_id.unique(),weight=weight)
        eval_df = pd.merge(y_pred, testset, how='inner', on=['user_id','movie_id'])
        #print(eval_df)
        print('find mse')
        mse = np.mean(np.power((eval_df['rating'] - eval_df['rating_pred']),2))
        rmse = np.sqrt(mse)
        return rmse
        

################################# Onestop SVD ##################################
def train(data):
    reader = Reader()
    svd = SVD()

    data_sp = Dataset.load_from_df(data[['user_id','movie_id','rating']], reader)
    train = data_sp.build_full_trainset()
    svd.fit(train)
    return svd
 
def predict(svd, user):
    rating_pred = [svd.predict(user, mid).est for mid in netflix_id.id]
    pred_user = pd.concat([netflix_id,pd.Series(rating_pred,name='rating_pred')], axis=1).sort_values('rating_pred', ascending=False)
    return pred_user
    
############################ Exploratory Data Analysis #########################
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

########################### Hypertuning SVD Parameters #########################
def hyperparameter_tuning(data):
    reader = Reader()
    data_rs = Dataset.load_from_df(data, reader)
    
    parameters = {'n_factors':[50,100,200],
                  'n_epochs':[20,40],
                  'lr_all':[0.005,0.001],
                  'reg_all':[0.05,0.02,0.01]}
    rs = RandomizedSearchCV(SVD, parameters, measures=['rmse'], cv=5)
    rs.fit(data_rs)
    
    return rs.best_params['rmse'], rs.best_score
    

########################### Prediction Visualization ###########################
def pred_visualization():
    model = load('..//trained//cf-svd.joblib')
    # randomly pick 230 users, they could be in the training set or new users
    pred = [{'user_id':i,'recommend': predict(model,i)} for i in range(1,100000000, 431823)]
    return pred


############################### Model Evaluation ###############################
def model_evaluation(data, method):
    
    assert((method == 'onestop') | (method == 'ensemble'))
    
    if method == 'onestop':
        svd = SVD(n_factors=50,
              n_epochs=20,
              lr_all=0.005,
              reg_all=0.05)
        
        reader = Reader()
        start = datetime.now()
        data_sp_cv = Dataset.load_from_df(data[['user_id','movie_id','rating']], reader)
        svd_cv = cross_validate(svd, data_sp_cv, measures=['rmse'],cv=5,verbose=True)
        print(datetime.now()-start)
    else:
        start = datetime.now()
        trainer = SVD_Ensemble()
        trainer.evaluate(data[0:1000000])
        print(datetime.now()-start)


def baseline_model_rmse(data):
    baseline = data['rating'].value_counts(normalize=True).sort_index()
    testset = data.sample(frac=0.2)
    pred = random.choices([1,2,3,4,5],baseline,k=len(testset))
    mse = np.mean(np.power((testset['rating'] - pred),2))
    rmse = np.sqrt(mse)
    print(rmse)
    


def main(method):
    
    assert((method == 'onestop') | (method == 'ensemble'))
    
    if method == 'onestop':
        svd = train(data)
        dump(svd, 'cf-svd.joblib')
    elif method == 'ensemble':
        trainer = SVD_Ensemble()
        start = datetime.now()
        trainer.fit(data)
        print(datetime.now()-start)
        for i in range(len(trainer.svds)):
            dump(trainer.svds[i], f'..//trained-ensemble//cf-svd-ensemble-{i}.joblib')

    weights = [0.3,0.5,0.7,0.9,1]
    start = datetime.now()
    rmses = {}
    trainer = SVD_Ensemble()
    for w in weights:
        print(w)
        rmses[w] = trainer.evaluate(data[0:1000000],weight=w)
        print(datetime.now()-start)
        



if __name__ == "__main__":
    
    exploratory(data)
    params = hyperparameter_tuning(data[['user_id','movie_id','rating']][0:1000000])

    baseline_model_rmse(data)

    main('onestop')
    
    pred = pred_visualization()

    model_evaluation(data[0:1000000],'onestop')
