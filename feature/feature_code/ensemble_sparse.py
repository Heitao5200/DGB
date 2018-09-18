# -*- coding: utf-8 -*-
"""
@简介：将data_w_tfidf(lda+lsa)+doc2vec特征转换为稀疏矩阵，并将其合并到tfidf
@author: Jian
"""
import pickle
from scipy import sparse
from scipy.sparse import hstack

"""=====================================================================================================================
0 读取data_w_tfidf(lda+lsa)+doc2vec特征
"""
print("0 读取data_w_tfidf(lda+lsa)+doc2vec特征")
f_ensemble = open('../feature_file/data_w_tfidf(lda+lsa)+doc2vec.pkl', 'rb')
x_train_ens, y_train, x_test_ens = pickle.load(f_ensemble)
f_ensemble.close()

"""=====================================================================================================================
1 将numpy 数组 转换为 csr稀疏矩阵
"""
print("1 将numpy 数组 转换为 csr稀疏矩阵")

x_train_ens_s = sparse.csr_matrix(x_train_ens)
x_test_ens_s = sparse.csc_matrix(x_test_ens)

"""=====================================================================================================================
2 读取tfidf特征
"""
print("2 读取tfidf特征")
f_tfidf = open('../feature_file/data_w_tfidf.pkl', 'rb')
x_train_tfidf, _, x_test_tfidf = pickle.load(f_tfidf)
f_tfidf.close()

"""=====================================================================================================================
3 对两个稀疏矩阵进行合并
"""
print("3 对两个稀疏矩阵进行合并")
x_train_spar = hstack([x_train_ens_s, x_train_tfidf])
x_test_spar = hstack([x_test_ens_s, x_test_tfidf])

"""=====================================================================================================================
4 将合并后的稀疏特征保存至本地
"""
data = (x_train_spar, y_train, x_test_spar)
f = open('../feature_file/data_w_tfidf+data_w_tfidf(lda+lsa)+doc2vec.pkl', 'wb')
pickle.dump(data, f)
f.close()




