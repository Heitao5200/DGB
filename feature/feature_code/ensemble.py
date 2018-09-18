# -*- coding: utf-8 -*-
"""
@brief : lda/lsa/doc2vec三种特征进行特征融合，并将结果保存至本地
@author: Jian
"""
import numpy as np
import pickle
import time

t_start = time.time()

"""=====================================================================================================================
2 读取[w]tfidf(lda)/tfidf(lsa)/doc2vec特征，并对这三种特征进行拼接融合
"""
print("2 读取[w]tfidf(lda)/tfidf(lsa)/doc2vec特征，并对这三种特征进行拼接融合")

f1 = open('../feature_file/data_w_doc2vec.pkl', 'rb')
x_train_1, y_train, x_test_1 = pickle.load(f1)
f1.close()


f2 = open('../feature_file/data_w_tfidf(lda).pkl', 'rb')
x_train_2, y_train, x_test_2 = pickle.load(f2)
f2.close()

f3 = open('../feature_file/data_w_tfidf(lsa).pkl', 'rb')
x_train_3, _, x_test_3 = pickle.load(f3)
f3.close()

x_train = np.concatenate((x_train_1, x_train_2, x_train_3), axis=1)
x_test = np.concatenate((x_test_1, x_test_2, x_test_3), axis=1)
# x_train = np.concatenate((x_train_1, x_train_2), axis=1)
# x_test = np.concatenate((x_test_1, x_test_2), axis=1)

"""=====================================================================================================================
2 将融合后的特征，保存至本地
"""
print("2 将融合后的特征，保存至本地")
data = (x_train, y_train, x_test)
fp = open('../feature_file/data_w_tfidf(lda+lsa)+doc2vec.pkl', 'wb')
pickle.dump(data, fp)
fp.close()