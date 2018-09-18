# -*- coding: utf-8 -*-
"""
@brief : 将tf特征降维为lda特征，并将结果保存至本地
@author: Jian
"""
from sklearn.decomposition import LatentDirichletAllocation
import pickle
import time

t_start = time.time()
# path = "E:/MyPython/机器学习——达观杯/feature/feature_file/"

"""=====================================================================================================================
1 tf特征加载
"""
print("1 tf特征加载")
f_tf = open('../feature_file/data_w_tf.pkl', 'rb')
x_train, y_train, x_test = pickle.load(f_tf)
f_tf.close()

"""=====================================================================================================================
2 特征降维：lda
"""
print("2 特征降维：lda")
lda = LatentDirichletAllocation(n_components=200)
x_train = lda.fit_transform(x_train)
x_test = lda.transform(x_test)

"""=====================================================================================================================
3 将lda特征保存至本地
"""
print("3 将lda特征保存至本地")
data = (x_train, y_train, x_test)
f_data = open('../feature_file/data_w_tf(lda).pkl', 'wb')
pickle.dump(data, f_data)
f_data.close()

t_end = time.time()
print("lda特征完成，共耗时：{}min".format((t_end-t_start)/60))


