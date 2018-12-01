# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 11:03:26 2018

@author: masha
"""

#若模型不保存直接预测的称为端到端
import numpy as np
#Dataframe manipulation library
import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
movies_df = pd.read_csv('H:download/ml-1m/movies.dat', sep='::', header=None,engine='python')
movies_df.head()
ratings_df = pd.read_csv('H:download/ml-1m/ratings.dat', sep='::', header=None,engine='python')
ratings_df.head()
movies_df.columns = ['MovieID', 'Title', 'Genres']
ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
merged_df = movies_df.merge(ratings_df, on='MovieID')
merged_df = merged_df.drop('Timestamp', axis=1).drop('Genres', axis=1)
userGroup1 = merged_df.sort_values('Rating',ascending=False).groupby('UserID')
tr_X=[]
for userID,curUser in userGroup1:
    a_new=curUser.sort_values('Rating',ascending=False)
    tr_X.append(list(a_new['Title']))
#训练过程
from gensim.models import Word2Vec
item2vec_model = Word2Vec(size=50, window=2, min_count=1, workers=2, sg=1)
#这里
item2vec_model.build_vocab (tr_X)
item2vec_model.train(tr_X,total_examples=len(tr_X),epochs=20)
item2vec_model.save ('updated_name_of_the_model')
#这个的目的是找到喜欢“Unforgiven (1992)”的类似的作品，把每个电影名称当做自然语言处理
#中的一个单词，（时序模型）
item2vec_model.most_similar('Unforgiven (1992)')
#保存模型之后加载model 并预测
item2vec_model1=Word2Vec.load("updated_name_of_the_model")
item2vec_model1.most_similar("Pump Up the Volume (1990)")