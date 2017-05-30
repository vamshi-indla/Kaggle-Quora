
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import timeit


# In[2]:

train_orig =  pd.read_csv('input/train.csv', header=0)
test_orig =  pd.read_csv('input/test.csv', header=0)


# In[3]:

tic0=timeit.default_timer()
df1 = train_orig[['question1']].copy()
df2 = train_orig[['question2']].copy()
df1_test = test_orig[['question1']].copy()
df2_test = test_orig[['question2']].copy()

# In[4]:

df2.rename(columns = {'question2':'question1'},inplace=True)
df2_test.rename(columns = {'question2':'question1'},inplace=True)

train_questions = df1.append(df2)
train_questions = train_questions.append(df1_test)
train_questions = train_questions.append(df2_test)
train_questions.drop_duplicates(subset = ['question1'],inplace=True)

len(train_questions)


# In[8]:

train_questions.reset_index(inplace=True,drop=True)

questions_dict = pd.Series(train_questions.index.values,index=train_questions.question1.values).to_dict()


# In[12]:

train_cp = train_orig.copy()
test_cp = test_orig.copy()
train_cp.drop(['qid1','qid2'],axis=1,inplace=True)


# In[13]:

test_cp['is_duplicate'] = -1
test_cp.rename(columns={'test_id':'id'},inplace=True)
comb = pd.concat([train_cp,test_cp])
comb.head(2)


# In[14]:

comb['q1_hash'] = comb['question1'].map(questions_dict)
comb['q2_hash'] = comb['question2'].map(questions_dict)



# In[16]:

q1_vc = comb.q1_hash.value_counts().to_dict()
q2_vc = comb.q2_hash.value_counts().to_dict()


# In[18]:

def try_apply_dict(x,dict_to_apply):
    try:
        return dict_to_apply[x]
    except KeyError:
        return 0
#map to frequency space
comb['q1_freq'] = comb['q1_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))
comb['q2_freq'] = comb['q2_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))


# In[20]:

train_comb = comb[comb['is_duplicate'] >= 0][['id','q1_hash','q2_hash','q1_freq','q2_freq','is_duplicate']]
test_comb = comb[comb['is_duplicate'] < 0][['id','q1_hash','q2_hash','q1_freq','q2_freq']]


# In[23]:
#more frequenct questions are more likely to be duplicates

corr_mat = train_comb.corr()
corr_mat.head()


# In[26]:
train_comb.to_csv('data/train_L0_Magic.csv', index=False)

# In[27]:
test_comb.to_csv('data/test_L0_Magic.csv', index=False)

#%%%

# Load L0 Clean features
train_Z = pd.read_csv("data/train_L0_clean.csv")
col = [c for c in train_Z.columns if c[:1] in ('x','y','z') ]

#%% Add new features to Old features and build model
fx_train = pd.DataFrame()
fx_train = train_comb[['q1_freq', 'q2_freq']]
#fx_train['q_freq'] = train_comb['q1_freq'] * train_comb['q2_freq']
fx_train[col] = train_Z[col]

fy_train = train_comb[['is_duplicate']]

#%%
# Split data into train and validation for a basic xgboost model building
from sklearn.cross_validation import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(fx_train, fy_train, test_size=0.2, random_state=4242)

#%%
# Xgboost
import xgboost as xgb

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)

#%% 
# Predict on test data
test_Z = pd.read_csv("data/test_L0_clean.csv",nrows=100)
x_test = test_comb[['q1_freq', 'q2_freq']]
x_test[col] = test_Z[col]
test_id = np.array(test_Z["test_id"])

xgtest = xgb.DMatrix(x_test)
preds = bst.predict(xgtest)

# Submission with L0 features + Magic feature
out_df = pd.DataFrame({"test_id":test_id, "is_duplicate":preds})
out_df.to_csv("pred_L0_Magic.csv", index=False)
print("Submission file created")

#%%
# Features importance
import operator
print("Features importances...")
importance = bst.get_fscore()
importance = sorted(importance.items(),key=operator.itemgetter(1))
ft = pd.DataFrame(importance, columns=['feature', 'fscore'])

ft.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
#plt.gcf().savefig('features_importance.png')
    

#%%
# Plot freq vs "is_Duplicate"
import matplotlib.pyplot as plt
plt.hist(fx_train['q_freq'],color = fy_train['is_duplicate'])
#plt.axis([0, 6, 0, 20])
plt.show()
