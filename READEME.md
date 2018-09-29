### 本项目为参加达观杯——文本分析过程的梳理

### 所用的数据集是从原始数据集中提取出来的，测试集和训练集各5000条

 1. 首先是数据处理  [提取5000条数据](data_processing/date_splite.py)
 2. 运行[第一个程序](for%20beginner/v0.0.py) 采用的是tfidf特征，SVM.LinearSVC模型
 3. 运行[第二个程序](for%20beginner/v0.1.py) 划分训练集和验证集，验证集比例为1：9
 4. 运行[第三个程序](for%20beginner/v0.2.py) 自动搜索分类器的最优的超参数值
 5. 运行[第四个程序](for%20beginner/v0.3.py) 自动搜索特征提取器和分类器的最优的超参数
 6. 运行[第五个程序](for%20beginner/v0.4.py) k折个模型进行融合。
 
### 跑过beginner里面的4程序后 应该整个流程有了一个大致的了解，后面提分的关键就是：
* 改变模型输入，即生成不同的特征[见feature](feature/feature_code)
* 尝试不同模型[见model](model/model_code)
* 模型融合
    * 概率文件融合([见proba](proba/proba_product.py))
    * 分类文件融合(参考概率文件融合)
    
**该比赛项目主要参考Jian老师的代码**

**感谢Jian老师,感谢小享，感谢一起战斗的队友!!!!**

 