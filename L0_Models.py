
import xgboost as xgb
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords

from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

# Load the data
train_Z = pd.read_csv("data/train_clean.csv",nrows=5000)

# check the dimensions of data
print(train.shape)
#print(test.shape)

# Check the first few rows
train.head()

# Xgboost function
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0):
        params = {}
        params["objective"] = "binary:logistic"
        params['eval_metric'] = 'logloss'
        params["eta"] = 0.05 # change learning rate
        params["subsample"] = 0.7
        params["min_child_weight"] = 1
        params["colsample_bytree"] = 0.7
        params["max_depth"] = 4
        params["silent"] = 1
        params["seed"] = seed_val
        num_rounds = 300 
        plst = list(params.items())
        xgtrain = xgb.DMatrix(train_X, label=train_y)

        if test_y is not None:
                xgtest = xgb.DMatrix(test_X, label=test_y)
                watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
                model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100, verbose_eval=10)
        else:
                xgtest = xgb.DMatrix(test_X)
                model = xgb.train(plst, xgtrain, num_rounds)
                
        pred_test_y = model.predict(xgtest)

        loss = 1
        if test_y is not None:
                loss = log_loss(test_y, pred_test_y)
                return pred_test_y, loss, model
        else:
            return pred_test_y, loss, model
        

#109]   train-logloss:0.353869  test-logloss:0.372451
#[107]   train-logloss:0.351571  test-logloss:0.369501 . + bi-grams
#[165]   train-logloss:0.344112  test-logloss:0.368647 . + no. of words diff 
#[196]   train-logloss:0.339208  test-logloss:0.36688 . + nouns
#[299]   train-logloss:0.313786  test-logloss:0.357048 . + POS #of words match
#[299]   train-logloss:0.302827  test-logloss:0.348327 . + POS Word Match ratio(no. of POS words removed)
#[299]   train-logloss:0.300021  test-logloss:0.344361 . + sentiment
#[299]   train-logloss:0.299681  test-logloss:0.344521 . + sentiment (retained valid chars only)
#[299]   train-logloss:0.341545  test-logloss:0.376738 only wordnet
    # Call Xgboost
#train_X = np.vstack( np.array(train.apply(lambda row: feature_extraction(row), axis=1)) ) 
#test_X = np.vstack( np.array(test.apply(lambda row: feature_extraction(row), axis=1)) )
col = [c for c in train_Z.columns if c[:] not in  ('is_duplicate' , 'question1' , 'question2')]

train_X = np.array(train_Z[col])
train_y = np.array(train_Z["is_duplicate"])

# separate the dups and non-dups
train_X_dup = train_X[train_y==1]
train_X_non_dup = train_X[train_y==0]
train_y_dup = train_y[train_y==1]
train_y_non_dup = train_y[train_y==0]

print(" Balance the Data ") 
train_X = np.vstack([train_X_non_dup, train_X_dup, train_X_non_dup, train_X_non_dup])
train_y = np.array([0]*train_X_non_dup.shape[0] + [1]*train_X_dup.shape[0] + [0]*train_X_non_dup.shape[0] + [0]*train_X_non_dup.shape[0])
del train_X_dup
del train_X_non_dup
print("Mean target rate : ",train_y.mean())
 

# 5 Fold Cross Validation
print(" Model Building Started ") 

kf = KFold(n_splits=5, shuffle=True, random_state=2016)
for dev_index, val_index in kf.split(range(train_X.shape[0])):
    print(len(dev_index))
    print(len(val_index))
    dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    preds, lloss, model = runXGB(dev_X, dev_y, val_X, val_y)
    break


print("Model Building is complete")

#test_Z = pd.read_csv("../Kaggle-Quora/input/test_clean.csv",nrows=5000)
#test_X  = np.array(test_Z[col])
#test_id = np.array(test_Z["test_id"])
#xgtest = xgb.DMatrix(test_X)
#preds = model.predict(xgtest)
