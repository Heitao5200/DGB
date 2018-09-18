import pandas as pd 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import xgboost as xgb

data_path = 'E:/MyPython/机器学习——达观杯/data_set/'
feature_path = 'E:/MyPython/机器学习——达观杯/feature/feature_file/'
proba_path = 'E:/MyPython/机器学习——达观杯/proba/proba_file/'
model_path = 'E:/MyPython/机器学习——达观杯/model/model_file/'
result_path ="E:/MyPython/机器学习——达观杯/result/"
print('读取特征:')
#读取Model
 
"""=====================================================================================================================
1 读取数据,并转换到XGB的标准数据格式
"""
print("1 读取数据,并转换到XGB的标准数据格式")
features_path = feature_path + 'data_w_tfidf.pkl'
data_fp = open(features_path, 'rb')
x_train, y_train, x_test = pickle.load(data_fp)
data_fp.close()

train_X,test_X, train_y, test_y = train_test_split(x_train,
                                                   y_train,
                                                   test_size = 0.2,
                                                   random_state = 0)


dtrain=xgb.DMatrix(train_X,label=train_y)
# for test
d_test = xgb.DMatrix(test_X)

# for save
dtest=xgb.DMatrix(x_test)

params={'booster':'gbtree',
    'objective': 'multi:softmax',
    'eval_metric': 'mlogloss',
    'max_depth':100,
    'lambda':10,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'min_child_weight':2,
    'eta': 0.05,
    'seed':0,
    'nthread':8,
     'silent':1,
 	'num_class':19}

watchlist = [(dtrain,'train')]

# ====================================训练
print('xgb训练')
bst = xgb.train(params,dtrain,num_boost_round=200,evals=watchlist)

# ====================================保存模型
joblib.dump(bst, model_path + "XGB_data_w_tfidf.m")

# ====================================report
y_pred_1 = bst.predict(d_test)

print('ACC: %.4f' % accuracy_score(test_y,y_pred_1))
print(classification_report(test_y,y_pred_1))


# ====================================预测并保存结果
y_pred = bst.predict(dtest)


# 保存
# df_test['class'] = y_pred.tolist()
# df_test['class'] = df_test['class'] + 1
# df_result = df_test.loc[:, ['id','class']]
df_result = pd.DataFrame(data={'id':range(5000), 'class': y_pred.tolist()})

df_result.to_csv(result_path +'XGB_data_w_tfidf.csv', index=False)











