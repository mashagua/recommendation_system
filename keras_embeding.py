# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 11:53:57 2018

@author: masha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import dummy, metrics, cross_validation, ensemble

import keras.models as kmodels
import keras.layers as klayers
import keras.backend as K
import keras
users = pd.read_csv('K:/Edownload/ml-1m/users.dat', sep='::', 
                        engine='python', 
                        names=['userid', 'gender', 'age', 'occupation', 'zip']).set_index('userid')
ratings = pd.read_csv('K:/Edownload/ml-1m/ratings.dat', engine='python', 
                          sep='::', names=['userid', 'movieid', 'rating', 'timestamp'])
movies = pd.read_csv('K:/Edownload/ml-1m/movies.dat', engine='python',
                         sep='::', names=['movieid', 'title', 'genre']).set_index('movieid')
movies['genre'] = movies.genre.str.split('|')
users.age = users.age.astype('category')
users.gender = users.gender.astype('category')
users.occupation = users.occupation.astype('category')
ratings.movieid = ratings.movieid.astype('category')
ratings.userid = ratings.userid.astype('category')

n_movies=movies.shape[0]
n_users=users.shape[0]
movieid=ratings.movieid.cat.codes.values
userid=ratings.userid.cat.codes.values
#把那个1000209个电影的打分转成one-hot类型
y=np.zeros((ratings.shape[0],5))
y[np.arange(ratings.shape[0]),ratings.rating-1]=1
####建立神经网络
movie_input=keras.layers.Input(shape=[1])
movie_vec=keras.layers.Flatten()(keras.layers.Embedding(n_movies+1,32)(movie_input))
movie_vec=keras.layers.Dropout(0.5)(movie_vec)

#同样的道理作用在user
user_input=keras.layers.Input(shape=[1])
user_vec=keras.layers.Flatten()(keras.layers.Embedding(n_users+1,32)(user_input))
user_vec=keras.layers.Dropout(0.5)(user_vec)
###user和movie concat
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
input_vecs = concatenate([movie_vec, user_vec])
#老方法，不能用
#input_vecs = keras.layers.merge([movie_vec, user_vec], mode='concat')
nn = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu')(input_vecs))
#加batchnormalization是为了加快收敛速度
nn = keras.layers.normalization.BatchNormalization()(nn)
nn = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu')(nn))
nn = keras.layers.normalization.BatchNormalization()(nn)
nn = keras.layers.Dense(128, activation='relu')(nn)

# Finally, we pull out the result!
result = keras.layers.Dense(5, activation='softmax')(nn)

# And make a model from it that we can actually run.
model = kmodels.Model([movie_input, user_input], result)
model.compile('adam', 'categorical_crossentropy')

# If we wanted to inspect part of the model, for example, to look
# at the movie vectors, here's how to do it. You don't need to 
# compile these models unless you're going to train them.
final_layer = kmodels.Model([movie_input, user_input], nn)
movie_vec = kmodels.Model(movie_input, movie_vec)
#分成测试集和验证集
a_movieid, b_movieid, a_userid, b_userid, a_y, b_y = cross_validation.train_test_split(movieid, userid, y)

history = model.fit([a_movieid, a_userid], a_y, 
                         epochs=10, 
                         validation_data=([b_movieid, b_userid], b_y))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
#对每个电影1-5分的概率
ss_test=model.predict([a_movieid, a_userid])
#真实的电影评分
s1=np.argmax(a_y,1)+1
#预测的电影评分
s2=np.argmax(ss_test, 1)+1
