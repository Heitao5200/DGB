print ("----------程序开始运行！！！------------")
import pickle
import pandas as pd
import time
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import  train_test_split
time_start = time.time()

data_path = 'E:/MyPython/机器学习——达观杯/data_set/'
feature_path = 'E:/MyPython/机器学习——达观杯/feature/feature_file/'
proba_path = 'E:/MyPython/机器学习——达观杯/proba/proba_file/'
model_path = 'E:/MyPython/机器学习——达观杯/model/model_file/'
result_path ="E:/MyPython/机器学习——达观杯/result/"
"""=====================================================================================================================
0 读取特征
"""
print("0 读取特征")
data_fp = open(feature_path  + "data_w_tfidf.pkl", 'rb')
x_train, y_train, x_test = pickle.load(data_fp)

xTrain, xTest, yTrain, yTest = train_test_split(x_train, y_train, test_size=0.30, random_state=531)

"""=====================================================================================================================
1 模型训练
"""
print("模型训练")
# clf = joblib.load('linearsvm_model_Tfid.1.m')
clf = svm.LinearSVC(C=5,dual=False)
clf.fit(x_train,y_train)

"""=====================================================================================================================
2 保存模型
"""
print('2 保存模型')
joblib.dump(clf, model_path + "SVM(c5)_data_w_tfidf.m")


"""=====================================================================================================================
3 预测结果 
"""
print("预测结果")
y_test = clf.predict(x_test)

"""=====================================================================================================================
4 保存结果 
"""
print("保存结果")
y_test = [i+1 for i in y_test.tolist()]
df_result = pd.DataFrame({'id':range(5000),'class':y_test})
df_result.to_csv(result_path + 'SVM(c5)_data_w_tfidf.csv',index=False)

time_end = time.time()
print('共耗时：{:.2f}min'.format((time_start-time_end)/60))