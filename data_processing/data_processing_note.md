# 本部分主要是对原始数据进行处理
## 读取文件的过程中遇到以下几个问题：

* 读取文件耗时长
    * 提取100条数据用来做测试 
    
* 读取文件报异常1：
```field larger than field limit (131072)```
```
import sys,csv
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
```

* 读取文件报异常2：OSError: Initializing from file failed
  
  读取文件加上参数 engine='python' 
  
  将读取文件引擎改为python （默认情况下是c）
    
* 原始数据的列名可能会出现乱码 故改列名
```
df_train.columns = ['id','article','word_seg',"class"]
df_test.columns = ['id','article','word_seg'] 
```
   
