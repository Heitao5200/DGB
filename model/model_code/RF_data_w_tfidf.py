# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 20:07:05 2018

@author: Simon
"""


print ("----------程序开始运行！！！------------")
import pickle
import pandas as pd
#from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import SGDClassifier
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.externals import joblib
import sys,csv
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import  train_test_split
from sklearn import ensemble
#from sklearn.ensemble import VotingClassifie
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.neural_network import MLPClassifier
time_start = time.time()
data_path = 'E:/MyPython/机器学习——达观杯/data_set/'
feature_path = 'E:/MyPython/机器学习——达观杯/feature/feature_file/'
proba_path = 'E:/MyPython/机器学习——达观杯/proba/proba_file/'
model_path = 'E:/MyPython/机器学习——达观杯/model/model_file/'
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
df_train=pd.read_csv(data_path + 'train_set1.csv',engine='python',encoding='gbk')
df_test=pd.read_csv(data_path + 'test_set1.csv',engine='python',encoding='gbk')
df_train.drop(df_train.columns[0],axis=1,inplace=True)

df_train["word_seg"] = df_train["article"].map(str) +' '+ df_train["word_seg"].map(str)
df_test["word_seg"] = df_test["article"].map(str) +' ' + df_test["word_seg"].map(str)

"""=====================================================================================================================
1 读取特征
"""
print('1 读取特征')
data_fp = open(feature_path  + "data_w_tfidf.pkl", 'rb')
x_train, y_train, x_test = pickle.load(data_fp)
xTrain, xTest, yTrain, yTest = train_test_split(x_train, y_train, test_size=0.30, random_state=531)

"""=====================================================================================================================
2 训练模型
"""
print('2 训练模型')
attackRFModel = ensemble.RandomForestClassifier(n_estimators=20, max_depth=None, max_features="auto",
                                               oob_score=False, random_state=531)
attackRFModel.fit(x_train,y_train)
prediction = attackRFModel.predict(xTest)
prediction2 = attackRFModel.predict(x_train)
correct = accuracy_score(yTest, prediction)
correct2 = accuracy_score(y_train, prediction2)
print(correct)
print(correct2)

"""=====================================================================================================================
3 保存模型
"""
print('3 保存模型')
joblib.dump(attackRFModel, model_path + "RF_data_w_tfidf.m")

"""=====================================================================================================================
4 预测结果 
"""
print("4 预测结果")
y_test = attackRFModel.predict(x_test)

"""=====================================================================================================================
5 保存结果 
"""
print("5 保存结果")
y_test = [i+1 for i in y_test.tolist()]
df_result = pd.DataFrame({'id':range(5000),'class':y_test})
df_result.to_csv(result_path + 'RF_data_w_tfidf.csv',index=False)
time_end = time.time()
print('共耗时：{:.2f}min'.format((time_start-time_end)/60))