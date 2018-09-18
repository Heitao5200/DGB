# -*- coding: utf-8 -*-
"""
@brief : LGB算法
@author: Jian
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import time
import pickle
import lightgbm as LGB
from sklearn.externals import joblib

t_start = time.time()
print("------------------程序开始---------------------")
data_path = 'E:/MyPython/机器学习——达观杯/data_set/'
feature_path = 'E:/MyPython/机器学习——达观杯/feature/feature_file/'
proba_path = 'E:/MyPython/机器学习——达观杯/proba/proba_file/'
model_path = 'E:/MyPython/机器学习——达观杯/model/model_file/'
result_path ="E:/MyPython/机器学习——达观杯/result/"

"""=====================================================================================================================
0 自定义验证集的评价函数
"""
print("0 自定义验证集的评价函数")
def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(19, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='macro')
    return 'f1_score', score_vali, True


"""=====================================================================================================================
1 读取数据,并转换到LGB的标准数据格式
"""
print("1 读取数据,并转换到LGB的标准数据格式")
data_fp = open(feature_path + 'data_w_tfidf.pkl' , 'rb')
x_train, y_train, x_test = pickle.load(data_fp)
data_fp.close()

"""=====================================================================================================================
2 划分训练集和验证集，验证集比例为test_size
"""
print("划分训练集和验证集，验证集比例为test_size")
x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train, test_size=0.1, random_state=0)
d_train = LGB.Dataset(data=x_train, label=y_train)
d_vali = LGB.Dataset(data=x_vali, label=y_vali)

"""=====================================================================================================================
3 训练LGB分类器
"""
print("3 训练LGB分类器")
params = {
    'boosting': 'gbdt',
    'application': 'multiclassova',
    'num_class': 19,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'max_depth': -1,
    'lambda_l1': 0,
    'lambda_l2': 0.5,
    'bagging_fraction': 1.0,

}

bst = LGB.train(params, d_train, num_boost_round=800, valid_sets=d_vali, feval=f1_score_vali,
                early_stopping_rounds=None,
                verbose_eval=True)

joblib.dump(bst, model_path + "LGB_data_w_tfidf.m")

"""=====================================================================================================================
4 对测试集进行预测;将预测结果转换为官方标准格式；并将结果保存至本地
"""
print("4 对测试集进行预测;将预测结果转换为官方标准格式；并将结果保存至本地")
y_proba = bst.predict(x_test)
y_test = np.argmax(y_proba, axis=1) + 1

df_result = pd.DataFrame(data={'id': range(5000), 'class': y_test.tolist()})
df_proba = pd.DataFrame(data={'id': range(5000), 'proba': y_proba.tolist()})

df_result.to_csv(result_path  + 'LGB_data_w_tfidf_result.csv', index=False)
df_proba.to_csv(result_path + 'LGB_data_w_tfidf_proba.csv', index=False)
t_end = time.time()
print("训练结束，耗时:{}min".format((t_end - t_start) / 60))


