# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 13:30:17 2018
@author: Simon
"""
import pickle
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import time
import sys,csv
from sklearn.calibration import CalibratedClassifierCV

data_path = 'E:/MyPython/机器学习————达观杯/data_set/'
feature_path = 'E:/MyPython/机器学习————达观杯/feature/feature_file/'
proba_path = 'E:/MyPython/机器学习————达观杯/proba/proba_file/'
model_path = 'E:/MyPython/机器学习————达观杯/model/model_file/'
result_path ="E:/MyPython/机器学习——达观杯/result/"

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
"""=====================================================================================================================
0 读取数据
"""
print('0 读取数据')
time_read_begin=time.time()
df_train=pd.read_csv(data_path +'train_set1.csv')
df_test=pd.read_csv(data_path +'test_set1.csv')
print (df_train.shape)
df_train=df_train[0:100]
df_train.drop(columns=['id','article'],inplace=True)
df_test.drop(columns=['article'],inplace=True)
#df_train.drop(columns=['id'],inplace=True)
#df_train["word_seg"] = df_train["article"].map(str) + df_train["word_seg"].map(str)
#df_test["word_seg"] = df_test["article"].map(str) + df_test["word_seg"].map(str)
time_read_end=time.time()
time_read=time_read_end-time_read_begin
print('1  read data end,time:'+ str(time_read))


"""=====================================================================================================================
1 读取特征
"""
print("1 读取特征")

data_fp = open(feature_path + "data_w_tfidf.pkl", 'rb')
x_train, y_train, x_test = pickle.load(data_fp)
#xTrain, xTest, yTrain, yTest = train_test_split(x_train, y_train, test_size=0.30, random_state=531)


"""=====================================================================================================================
2 读取model+预测
"""

print('1.1 读取svm_model+预测概率')
svm = joblib.load("svm(c5).pkl")
clf1 = CalibratedClassifierCV(base_estimator=svm,  cv ="prefit" )
clf1.fit(x_train,y_train)
y_test = clf1.predict_proba(x_test)
df_test['proba']=y_test.tolist()
df_result = df_test.loc[:,['id','proba']]
df_result.to_csv('result_proba_svm.csv',index=False)

print('1.2 读取lr_model+预测概率')
clf2 = joblib.load("lr(c40).pkl")
y_test=clf2.predict_proba(x_test)
df_test['proba']=y_test.tolist()
df_result = df_test.loc[:,['id','proba']]
df_result.to_csv('result_proba_lg.csv',index=False)

print('1.3 读取kn_model+预测概率')
clf3 = joblib.load("kn.pkl")
y_test=clf2.predict_proba(x_test)
df_test['proba']=y_test.tolist()
df_result = df_test.loc[:,['id','proba']]
df_result.to_csv('result_proba_kn.csv',index=False)

print('1.4 读取nb_model+预测概率')
clf4 = joblib.load("nb_try.m")
y_test=clf2.predict_proba(x_test)
df_test['proba']=y_test.tolist()
df_result = df_test.loc[:,['id','proba']]
df_result.to_csv('result_proba_nb.csv',index=False)






print('4 读取概率+投票')
svm_df = pd.read_csv('result_proba_svm.csv')
lg_df = pd.read_csv('result_proba_lg.csv')
kn_df = pd.read_csv('result_proba_kn.csv')
nb_df = pd.read_csv('result_proba_nb.csv')
def series2arr(series):
    res = []
    for row in series:
        res.append(np.array(eval(row)))
    return np.array(res)

# Series
svm_prob_arr = series2arr(svm_df['proba'])
lg_prob_arr = series2arr(lg_df['proba'])
kn_prob_arr = series2arr(kn_df['proba'])
nb_prob_arr = series2arr(nb_df['proba'])

final_prob = svm_prob_arr+lg_prob_arr+kn_prob_arr+nb_prob_arr

y_class=[np.argmax(row) for row in final_prob]

df_test['proba']=y_class
df_test['proba']=df_test['proba']+1
df_result = df_test.loc[:,['id','proba']]


print('5  save predictable data')
df_result.to_csv('result.csv',index=False)