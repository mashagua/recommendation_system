# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 10:54:25 2018

@author: masha
"""

import tensorflow as tf
#Numpy contains helpful functions for efficient mathematical calculations
import numpy as np
#Dataframe manipulation library
import pandas as pd
#Graph plotting library
import matplotlib.pyplot as plt
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
%matplotlib inline
#Loading in the movies dataset
movies_df = pd.read_csv('H:download/ml-1m/movies.dat', sep='::', header=None,engine='python')
movies_df.head()
ratings_df = pd.read_csv('H:download/ml-1m/ratings.dat', sep='::', header=None,engine='python')
ratings_df.head()
movies_df.columns = ['MovieID', 'Title', 'Genres']
ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
#同一个movie_id可以对应不同的index
movies_df['List Index'] = movies_df.index
movies_df.head()
merged_df = movies_df.merge(ratings_df, on='MovieID')
merged_df = merged_df.drop('Timestamp', axis=1).drop('Genres', axis=1)
userGroup = merged_df.groupby('UserID')
trX=[]
# For each user in the group
for userID, curUser in userGroup:
    # Create a temp that stores every movie's rating
    temp = [0]*len(movies_df)
    # For each movie in curUser's movie list
    for num, movie in curUser.iterrows():
        # Divide the rating by 5 and store it
        temp[movie['List Index']] = movie['Rating']/5.0
    # Add the list of ratings into the training list
    trX.append(temp)

hiddenUnits = 20
visibleUnits = len(movies_df)
vb = tf.placeholder("float", [visibleUnits]) #Number of unique movies
hb = tf.placeholder("float", [hiddenUnits]) #Number of features we're going to learn
W = tf.placeholder("float", [visibleUnits, hiddenUnits])
v0=tf.placeholder("float",[None,visibleUnits])
_h0=tf.nn.sigmoid(tf.matmul(v0,W)+hb)
h0=tf.nn.relu(tf.sign(_h0-tf.random_uniform(tf.shape(_h0))))
###
_v1=tf.nn.sigmoid(tf.matmul(h0,tf.transpose(W))+vb)
v1=tf.nn.relu(tf.sign(_v1-tf.random_uniform(tf.shape(_v1))))
h1=tf.nn.sigmoid(tf.matmul(v1, W)+hb)
alpha=1.0
w_pos_grad=tf.matmul(tf.transpose(v0),h0)
w_neg_grad=tf.matmul(tf.transpose(v1),h1)
CD=(w_pos_grad-w_neg_grad)/tf.to_float(tf.shape(v0)[0])
update_w=W+alpha*CD
#按照行求平均
update_vb=vb+alpha*tf.reduce_mean(v0-v1,0)
update_hb=hb+alpha*tf.reduce_mean(h0-h1,0)
#根据错误调整隐含层的权重
err=v0-v1
err_sum=tf.reduce_mean(err*err)
cur_w=np.zeros([visibleUnits,hiddenUnits],np.float32)
cur_vb=np.zeros([visibleUnits],np.float32)
cur_hb=np.zeros([hiddenUnits],np.float32)
####
prev_w=np.zeros([visibleUnits,hiddenUnits],np.float32)
prev_vb=np.zeros([visibleUnits],np.float32)
prev_hb=np.zeros([hiddenUnits],np.float32)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
epochs=15
batchsize=100
errors=[]
for i in range(epochs):
    for start,end in zip(range(0,len(trX),batchsize),range(batchsize,len(trX),batchsize)):
        batch=trX[start:end]
        cur_w=sess.run(update_w,feed_dict={v0:batch,W:prev_w,vb:prev_vb,hb:prev_hb})
        cur_vb=sess.run(update_vb,feed_dict={v0:batch,W:prev_w,vb:prev_vb,hb:prev_hb})
        cur_hb=sess.run(update_hb,feed_dict={v0:batch,W:prev_w,vb:prev_vb,hb:prev_hb})
        prev_w=cur_w
        prev_vb=cur_vb
        prev_hb=cur_hb
    errors.append(sess.run(err_sum,feed_dict={v0: trX, W: cur_w, vb: cur_vb, hb: cur_hb}))
    print(errors[-1])
plt.plot(errors)
plt.ylabel('error')
plt.xlabel('epoch')
plt.show()
inputUser=[trX[50]]
[x for x in range(len(trX[50])) if trX[50][x]!=0]
hh0=tf.nn.sigmoid(tf.matmul(v0,W)+hb)
vv1=tf.nn.sigmoid(tf.matmul(hh0,tf.transpose(W))+vb)
feed=sess.run(hh0,feed_dict={v0:inputUser,W:prev_w,hb:prev_hb})
rec=sess.run(vv1,feed_dict={hh0:feed,W:prev_w,vb:prev_vb})
scored_movies_df_50=movies_df
scored_movies_df_50["Recommendation Score"]=rec[0]
scored_movies_df_50.sort_values(["Recommendation Score"],ascending=False).head(20)
movies_df_50=merged_df[merged_df['UserID']==merged_df.iloc[50]['UserID']]
movies_df_50.head()
merged_df_50=scored_movies_df_50.merge(movies_df_50,on='MovieID',how="outer")
merged_df_50=merged_df_50.drop("List Index_y",axis=1).drop("UserID",axis=1)
rs1=merged_df_50.sort_values(["Recommendation Score"], ascending=False)
#推荐的电影
rs1_new=rs1.loc[rs1['Rating'].isnull()]
print(rs1_new.head())
