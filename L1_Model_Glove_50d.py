"""
Created on Tue May  9 06:17:34 2017

@author: vamshi294
# Glove with 50D
"""

import pandas as pd
import numpy as np
np.random.seed(1)

import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages')
sys.path.append('/Users/vamshi294/Documents/Text Analytics/Kaggle_Quora')

from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Merge
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text
from string import punctuation
from nltk.corpus import stopwords
stop_punctuation = set(punctuation)
stopwords = set(stopwords.words('english'))
stop_words = set(["does","do","i","you","your","yours","to","this","the","a","in",
"and","it","on","of","u","so","for","we","if","by","as","me"])
stop_words = set(stop_words) | set(punctuation)

import os
os.chdir("/Users/vamshi294/Documents/Text Analytics/Kaggle_Quora/Kaggle-Quora")

#%%
embeddings_index = {}
f = open('data/glove.6B.50d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    try:
        coefs = np.asarray(values[1:], dtype='float32')
    except ValueError:
        continue
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

#%%
from nltk import word_tokenize
import re
import enchant
d = enchant.Dict("en_US")

def str_stem(s): 
    if isinstance(s, str):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
 
        # short forms
        s = re.sub("what's","what is", s)
        s = re.sub("what're","what are", s)
        s = re.sub("who's","who is", s)
        s = re.sub("who're","who are", s)
        s = re.sub("where's","where is", s)
        s = re.sub("where're","where are", s)
        s = re.sub("when's","when is", s)
        s = re.sub("when're","when are", s)
        s = re.sub("how's","how is", s)
        s = re.sub("how're","how are", s)
        s = re.sub("i'm","i am", s)
        s = re.sub("we're","we are", s)
        s = re.sub("you're","you are", s)
        s = re.sub("they're","they are", s)
        s = re.sub("it's","it is", s)
        s = re.sub("he's","he is", s)
        s = re.sub("she's","she is", s)
        s = re.sub("that's","that is", s)
        s = re.sub("there's","there is", s)
        s = re.sub("there're","there are", s)
    
        s = re.sub("i've","i have", s)
        s = re.sub("we've","we have", s)
        s = re.sub("you've","you have", s)
        s = re.sub("they've","they have", s)
        s = re.sub("who've","who have", s)
        s = re.sub("would've","would have", s)
        s = re.sub("not've","not have", s)
    
        s = re.sub("i'll","i will", s)
        s = re.sub("we'll","we will", s)
        s = re.sub("you'll","you will", s)
        s = re.sub("he'll","he will", s)
        s = re.sub("she'll","she will", s)
        s = re.sub("it'll","it will", s)
        s = re.sub("they'll","they will", s)
    
        s = re.sub("isn't","is not", s)
        s = re.sub("wasn't","was not", s)
        s = re.sub("aren't","are not", s)
        s = re.sub("weren't","were not", s)
        s = re.sub("can't","can not", s)
        s = re.sub("couldn't","could not", s)
        s = re.sub("don't","do not", s)
        s = re.sub("didn't","did not", s)
        s = re.sub("shouldn't","should not", s)
        s = re.sub("wouldn't","would not", s)
        s = re.sub("doesn't","does not", s)
        s = re.sub("haven't","have not", s)
        s = re.sub("hasn't","has not", s)
        s = re.sub("hadn't","had not", s)
        s = re.sub("won't","will not", s)

        s = re.sub(r"(whatis|Whatis)", "what is ", s)
        s = re.sub(r" u ", " you ", s)
        s = re.sub(r" ur ", " your ", s)
        s = re.sub(r"0k", "0000 ", s) #check the numbers conversion

        
        # Spell corrections
        s = re.sub(r"actived", "active", s)
        s = re.sub(r"programing", "programming", s)
        s = re.sub(r"calender", "calendar", s)
        s = re.sub("favourite","favorite",s)
        s = re.sub("travelling"," traveling",s)
        s = re.sub("quikly","quickly",s)
        s = re.sub(r" uk ", " england ", s)
        s = re.sub(r"imrovement", "improvement", s)
        s = re.sub(r"intially", "initially", s)
        s = re.sub(r" dms ", "direct messages ", s)
        s = re.sub(r"demonitization", "demonetization", s)
        s = re.sub(" colour "," color ", s)
        s = re.sub("\0rs"," rs", s)
       
        # Full forms
        s = re.sub(r"( *)(US|u s|USA|the us)( *)", r" america ", s)  
        s = re.sub(r"( *)(Upsc|UPSC|upsc)( *)", r" civil services ", s)        
        s = re.sub(r"kms", " kilometers ", s)
        s = re.sub(r" cs ", " computer science ", s)
        s = re.sub(r"( *)(cse|CSE)( *)", r" computer science engineering ", s)
        s = re.sub(r"( *)(ece|ECE)( *)", r" electronics and communications engineering ", s)
        s = re.sub(r"( *)(eee|EEE)( *)", r" electrical and electronics engineering ", s)
        s = re.sub(r"( *)(tv|TV)( *)", r"television", s)

        #Standardize
        s = re.sub(r"( +)(INR|inr|rs|Rs)( *)", r" rupees ", s)
        s = re.sub(r"( *)(india|INDIA|India)( *)", r" India ", s)
        s = re.sub(r"( *)(wifi|Wi Fi)( *)", r"WiFi", s)
        s = re.sub(r"bestfriend", "best friend", s)
        s = s.replace(" cambodia "," Cambodia ")
        s = s.replace(" pune "," Pune ")
        s = s.replace(" youtube "," Youtube ")
        s = s.replace(" Obama "," Barack Obama ")
        s = s.replace(" behaviour "," behavior ")
        s = s.replace(" gmail "," Gmail ")
        s = s.replace(" whatsapp "," Whatsapp ")
        s = s.replace(" apps "," application ")
        s = s.replace(" app "," application ")
        s = s.replace(" Modi "," Narendra Modi ")
        s = s.replace(" ok "," OK ")
        s = re.sub(r"[\,\''\``\-\/\“\”]", " ", s)        
        s = re.sub(r"\^[A-Za-z0-9\']", " ", s)
        
        s = word_tokenize(s)
        s = " ".join([w for w in s if not w.lower() in stop_words])

        return s
    else:
        return "null"

def str_format(s):
    commonwords = set(str(s['question1']).lower().split()).intersection(set(str(s['question2']).lower().split()))
    if len(commonwords) > 0:
        for y in commonwords:
            for q1 in s['question1'].split():
                if str(q1).lower() == y:
                    s['question1'] = str(s['question1']).replace(str(q1),str(y))
                    
            for q2 in s['question2'].split():
                if str(q2).lower() == y:
                    s['question2'] = str(s['question2']).replace(str(q2),str(y))    
        return s
    else:
        return s
    
#%%
#Load the data into array and convert to word_index
data = pd.read_csv("input/train.csv",nrows=5000)

y = data.is_duplicate.values

#%% 
#Clean the data
data['question1'] = data.question1.apply(lambda x: str_stem(x))
data['question2'] = data.question2.apply(lambda x: str_stem(x))
data  = data.apply(lambda x: str_format(x),axis=1)

#%% Convert to word_index
       
tk = text.Tokenizer(num_words=200000)

max_len = 40
dim = 50 #50 dim array
tk.fit_on_texts(list(data.question1.values.astype(str)) + list(data.question2.values.astype(str)))
word_index = tk.word_index

#print(tk.word_index)
x1 = tk.texts_to_sequences(data.question1.values.astype(str))
x1 = sequence.pad_sequences(x1, maxlen=max_len)

x2 = tk.texts_to_sequences(data.question2.values.astype(str))
x2 = sequence.pad_sequences(x2, maxlen=max_len)

#%%
# convert text to embedded vectors
from spell import *

temp_text3 = " "
embedding_matrix = np.zeros((len(word_index) + 1, dim))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is None:      
        word = correction(word)
        embedding_vector = embeddings_index.get(word)
    else:
        temp_text3 = temp_text3 + " " + str(word)                    
        
"""
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        temp_text3 = temp_text3 + " " + str(word) 
"""                   
#%%
#max_features = 200000
filter_length = 5
nb_filter = 64
pool_length = 4

model = Sequential()
print('Build model...')

#===============
model1 = Sequential()
model1.add(Embedding(len(word_index) + 1,
                     dim,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))

model1.add(TimeDistributed(Dense(dim, activation='relu')))
model1.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(dim,)))

#===============
model2 = Sequential()
model2.add(Embedding(len(word_index) + 1,
                     dim,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))

model2.add(TimeDistributed(Dense(dim, activation='relu')))
model2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(dim,)))
#===============

model3 = Sequential()
model3.add(Embedding(len(word_index) + 1,
                     dim,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model3.add(Convolution1D(filters=nb_filter,
                         kernel_size=filter_length,
                         padding='valid',
                         activation='relu',
                         strides=1))
model3.add(Dropout(0.2))

model3.add(Convolution1D(filters=nb_filter,
                         kernel_size=filter_length,
                         padding='valid',
                         activation='relu',
                         strides=1))

model3.add(GlobalMaxPooling1D())
model3.add(Dropout(0.2))

model3.add(Dense(dim))
model3.add(Dropout(0.2))
model3.add(BatchNormalization())

#===============

model4 = Sequential()
model4.add(Embedding(len(word_index) + 1,
                     dim,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model4.add(Convolution1D(filters=nb_filter,
                         kernel_size=filter_length,
                         padding='valid',
                         activation='relu',
                         strides=1))
model4.add(Dropout(0.2))

model4.add(Convolution1D(filters=nb_filter,
                         kernel_size=filter_length,
                         padding='valid',
                         activation='relu',
                         strides=1))

model4.add(GlobalMaxPooling1D())
model4.add(Dropout(0.2))

model4.add(Dense(dim))
model4.add(Dropout(0.2))
model4.add(BatchNormalization())
#===============

model5 = Sequential()
model5.add(Embedding(len(word_index) + 1, dim, input_length=max_len))
model5.add(LSTM(dim, dropout=0.2, recurrent_dropout=0.2))
#===============

model6 = Sequential()
model6.add(Embedding(len(word_index) + 1, dim, input_length=max_len))
model6.add(LSTM(dim, dropout=0.2, recurrent_dropout=0.2))
#===============

merged_model = Sequential()
merged_model.add(Merge([model1, model2, model3, model4, model5, model6], mode='concat'))
merged_model.add(BatchNormalization())

merged_model.add(Dense(dim))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(dim))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(dim))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(dim))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(dim))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(1))
merged_model.add(Activation('sigmoid'))
#===============

#%%
# Model fitting
merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpoint = ModelCheckpoint('weights.h6', monitor='val_acc', save_best_only=True, verbose=2)

merged_model.fit([x1, x2, x1, x2, x1, x2], y=y, batch_size=5000, epochs=100,
                 verbose=1, validation_split=0.2, shuffle=False, callbacks=[checkpoint])

#%%
# Evaluate model
scores = merged_model.evaluate([x1, x2, x1, x2, x1, x2], y=y)
print("\n%s: %.2f%%" % (merged_model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = merged_model.predict([x1, x2, x1, x2, x1, x2])

# round predictions
rounded = [round(x[0]) for x in predictions]
#print(rounded)

from sklearn.metrics import confusion_matrix
confusion_matrix(y,rounded)
#72

#%%
# Predict on Test data

#Load the data into array and convert to word_index
test = pd.read_csv("input/test.csv",nrows=10)

test['question1'] = test.question1.apply(lambda x: str_stem(x))
test['question2'] = test.question2.apply(lambda x: str_stem(x))
test  = test.apply(lambda x: str_format(x),axis=1)

#tk.fit_on_texts(list(test.question1.values.astype(str)) + list(test.question2.values.astype(str)))
#word_index = tk.word_index
z1 = tk.texts_to_sequences(test.question1.values.astype(str))
z1 = sequence.pad_sequences(z1, maxlen=max_len)

z2 = tk.texts_to_sequences(test.question2.values.astype(str))
z2 = sequence.pad_sequences(z2, maxlen=max_len)

# calculate predictions
predictions_test = merged_model.predict([z1, z2, z1, z2, z1, z2])

# round predictions
rounded_test = [round(x[0]) for x in predictions_test]

confusion_matrix(test.is_duplicate.values,rounded_test)
#%%
# predict on Test data
out_df = pd.DataFrame({"test_id":np.array(test["test_id"]), "is_duplicate":rounded_test})
out_df.to_csv("pred_glove_50d.csv", index=False)
print("Submission file created")


"""
# References

Aws + anaconda setup for competing in Kaggle:
http://www.grant-mckinnon.com/?p=6
http://www.grant-mckinnon.com/?p=56

# Best collection of kernels for quora question pair
https://www.kaggle.com/c/quora-question-pairs/discussion/32819

# Install Anacondas with python 3.6
wget http://repo.continuum.io/archive/Anaconda3-4.3.0-Linux-x86_64.sh
bash Anaconda3-4.3.0-Linux-x86_64.sh

export PATH=/home/ubuntu/anaconda3/bin:$PATH
source .bashrc
which python

#set password
from  IPython.lib import passwd
passwd()

jupyter notebook --generate-config

mkdir certs
cd certs
sudo openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout mycert.pem -out mycert.pem


cd ~/.jupyter/
vi jupyter_notebook_config.py



cd ~
mkdir Notebooks
cd Notebooks
jupyter notebook

# Open the browser and provide the DNS


# Move files to AWS
scp -i /Users/vamshi294/Documents/'Text Analytics'/Kaggle_Quora/Kaggle-Quora/kaggletutorial1.pem /Users/vamshi294/Documents/'Text Analytics'/Kaggle_Quora/Kaggle-Quora/input/train.csv ubuntu@ec2-54-209-59-224.compute-1.amazonaws.com:~
scp -i /Users/vamshi294/Documents/'Text Analytics'/Kaggle_Quora/Kaggle-Quora/kaggletutorial1.pem /Users/vamshi294/Documents/'Text Analytics'/Kaggle_Quora/Kaggle-Quora/input/test.csv ubuntu@ec2-54-209-59-224.compute-1.amazonaws.com:~

#ssh 
ssh -i "kaggletutorial1.pem" ubuntu@ec2-54-209-59-224.compute-1.amazonaws.com


"""