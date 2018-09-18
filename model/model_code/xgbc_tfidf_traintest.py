### 修改中
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.metrics import accuracy_score
import time
import pickle

data_path = 'E:/MyPython/机器学习——达观杯/data_set/'
feature_path = 'E:/MyPython/机器学习——达观杯/feature/feature_file/'
proba_path = 'E:/MyPython/机器学习——达观杯/proba/proba_file/'
model_path = 'E:/MyPython/机器学习——达观杯/model/model_file/'
result_path ="E:/MyPython/机器学习——达观杯/result/"
## 抽样
print('读取特征:')
#读取Model
features_path = feature_path + 'data_w_tfidf.pkl'
data_fp = open(features_path, 'rb')
x_train, y_train, x_test = pickle.load(data_fp)
data_fp.close()

X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.02, random_state=0)

print('开始训练:')
start = time.time() 

# max_depth=15, n_estimators=30, learning_rate=0.05
# 这里还可以尝试其他多种模型，利用fit()函数和预测predict(),这里使用XGboost
gbm = xgb.XGBClassifier().fit(X_train, y_train)
print('训练集上的误差：{}'.format(gbm.score(X_train,y_train)))


pred_test = gbm.predict(X_test)
accuracy = accuracy_score(y_test, pred_test)
print("测试集上的准确率:{}".format(accuracy))

print("测试集报告")
print(classification_report(y_test, pred_test))

end = time.time()
print('time',end-start)



y_test = gbm.predict(x_test)

df_result = pd.DataFrame(data={'id':range(102277), 'class': y_pred.tolist()})

df_result.to_csv(result_path +'XGB_data_w_tfidf.csv', index=False)