'''
由于原始数据集太大，读取耗时长，
占用电脑太多内存资源，从原始数据中提取1000条数据
'''
from __future__ import print_function
import sys,csv
import pandas as pd
import time
start_read_time=time.time()
path = "E:/MyPython/CupContest/数据集/"
print('————————————————————读取数据———————————————————————————————')

'''
在读取文件的数据时，偶尔会抛出异常：_csv.Error: field larger than field limit (131072)
还有一个异常：OverflowError: Python int too large to convert to C long
可以看出报错的原因是读取的文件数据太大，加上下面代码后，数据能够正常读取
'''

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

"""
文件切分
"""
df_train = pd.read_csv( path + 'train_set.csv',nrows=5000,engine='python',encoding='gbk')
df_test = pd.read_csv(path + 'test_set.csv',nrows=5000,engine='python',encoding='gbk')

"""
原始数据的列名可能会出现乱码
故改列名
"""
df_train.columns = ['id','article','word_seg',"class"]
df_test.columns = ['id','article','word_seg']

"""
保存文件
"""
df_train.to_csv(path + 'train_set1.csv',index=False)
df_test.to_csv(path + 'test_set1.csv',index=False)

end_read_time=time.time()
print("—————————————读取数据结束，耗时:%.2fs————————————————————————"%(end_read_time-start_read_time))