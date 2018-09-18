# -*- coding: utf-8 -*-
"""
@brief : 将原始数据数字化为doc2vec特征，并将结果保存至本地
@author: Jian
"""
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import time
import pickle
import sys,csv

t_start = time.time()
maxInt = sys.maxsize
decrement = True
while decrement:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.
    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True
path = "E:/MyPython/CupContest/数据集/"
"""=====================================================================================================================
0 辅助函数 
"""
#将句子切分 转化为list
def sentence2list(sentence):
    s_list = sentence.strip().split()
    return s_list


"""=====================================================================================================================
1 读取原始数据，并进行简单处理
"""
print("1 读取原始数据，并进行简单处理")
df_train = pd.read_csv( '../../data_set/train_set1.csv',engine='python')
df_test = pd.read_csv('../../data_set/test_set1.csv',engine='python')

df_train.drop(columns='article', inplace=True)
df_test.drop(columns='article', inplace=True)

df_all = pd.concat(objs=[df_train, df_test], axis=0, sort=True)
y_train = (df_train['class'] - 1).values

df_all['word_list'] = df_all['word_seg'].apply(sentence2list)
texts = df_all['word_list'].tolist()

"""=====================================================================================================================
2 doc2vec
"""
print("2 doc2vec")
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
# model = Doc2Vec(documents, vector_size=200, window=5, min_count=3, workers=4, epochs=25)
model = Doc2Vec(documents, window=5, min_count=3, workers=4)
docvecs = model.docvecs

x_train = []
for i in range(0, 5000):
    x_train.append(docvecs[i])
x_train = np.array(x_train)

x_test = []
for j in range(5000, 10000):
    x_test.append(docvecs[j])
x_test = np.array(x_test)

"""=====================================================================================================================
3 将doc2vec特征保存至本地
"""
print("3 将doc2vec特征保存至本地")
data = (x_train, y_train, x_test)
f_data = open('E:/MyPython/机器学习——达观杯/feature/feature_file/data_w_doc2vec.pkl', 'wb')
pickle.dump(data, f_data)
f_data.close()

t_end = time.time()
print("已将原始数据数字化为doc2vec特征，共耗时：{}min".format((t_end-t_start)/60))
