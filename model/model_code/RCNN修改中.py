##  修改中
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras import backend
backend.clear_session()
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.utils.np_utils import to_categorical
from keras import initializers
#from keras import backend as K
from keras.engine.topology import Layer
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

## 先把原始的文本处理成2000维的向量，太长的截断，不够的补0
## 生成300维的嵌入
## CNN，3个256的卷积，池化以后，flatten，输入给softmax
## 输出分类的one hot编码


### 原始输入
# ~/Downloads/train_set.csv
# ~/workspace/sublime/daguan/train_sample.csv
data_path = 'E:/MyPython/机器学习——达观杯/data_set/'
feature_path = 'E:/MyPython/机器学习——达观杯/feature/feature_file/'
proba_path = 'E:/MyPython/机器学习——达观杯/proba/proba_file/'
model_path = 'E:/MyPython/机器学习——达观杯/model/model_file/'
result_path ="E:/MyPython/机器学习——达观杯/result/"
#path='D:/DGB/new_data/'

doc_len = 2000
embedding_dim = 300


print('read data')

df_train = pd.read_csv(data_path  + 'train_set1.csv',engine='python',encoding='gbk')
df_test = pd.read_csv(data_path  + 'train_set1.csv',engine='python',encoding='gbk')
print (df_train.shape)
#df_train=df_train[0:500]
df_train.drop(df_train.columns[0],axis=1,inplace=True)
#df_train["word_seg"] = df_train["article"].map(str) +' '+ df_train["word_seg"].map(str)
#df_test["word_seg"] = df_test["article"].map(str) +' ' + df_test["word_seg"].map(str)


word_seg = df_train['word_seg']
label = df_train['class'] - 1
X_train, X_test, y_train, y_test = train_test_split(word_seg, label, test_size=0.1, random_state=42)

print('embedding')
y_labels = list(y_train.value_counts().index)
le = preprocessing.LabelEncoder()
le.fit(y_labels)
num_labels = len(y_labels)
y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), num_labels)
y_test = to_categorical(y_test.map(lambda x: le.transform([x])[0]), num_labels)


tokenizer = Tokenizer(split=' ')
tokenizer.fit_on_texts(word_seg)
vocab = tokenizer.word_index

# pad是填充，意思是在前面补零，处理完后长度均为200
# 输入转换

X_train_word_ids = tokenizer.texts_to_sequences(X_train)
X_test_word_ids = tokenizer.texts_to_sequences(X_test)

X_train_padded_seqs = pad_sequences(X_train_word_ids, maxlen=doc_len)
X_test_padded_seqs = pad_sequences(X_test_word_ids, maxlen=doc_len)

left_train_word_ids = [[len(vocab)] + x[:-1] for x in X_train_word_ids]
left_test_word_ids = [[len(vocab)] + x[:-1] for x in X_test_word_ids]
right_train_word_ids = [x[1:] + [len(vocab)] for x in X_train_word_ids]
right_test_word_ids = [x[1:] + [len(vocab)] for x in X_test_word_ids]

# 分别对左边和右边的词进行编码
left_train_padded_seqs = pad_sequences(left_train_word_ids, maxlen=doc_len)
left_test_padded_seqs = pad_sequences(left_test_word_ids, maxlen=doc_len)
right_train_padded_seqs = pad_sequences(right_train_word_ids, maxlen=doc_len)
right_test_padded_seqs = pad_sequences(right_test_word_ids, maxlen=doc_len)

# 模型共有三个输入，分别是左词，右词和中心词
document = Input(shape = (doc_len, ), dtype = "int32")
left_context = Input(shape = (doc_len, ), dtype = "int32")
right_context = Input(shape = (doc_len, ), dtype = "int32")

# 构建词向量
embedder = Embedding(len(vocab) + 1, embedding_dim, input_length = doc_len)
doc_embedding = embedder(document)
l_embedding = embedder(left_context)
r_embedding = embedder(right_context)

# 分别对应文中的公式(1)-(7)
print('model')
forward = LSTM(256, return_sequences = True)(l_embedding) # 等式(1)
# 等式(2)
backward = LSTM(256, return_sequences = True, go_backwards = True)(r_embedding)
together = concatenate([forward, doc_embedding, backward], axis = 2) # 等式(3)

semantic = TimeDistributed(Dense(128, activation = "tanh"))(together) # 等式(4)
# 等式(5)
pool_rnn = Lambda(lambda x: backend.max(x, axis = 1), output_shape = (128, ))(semantic)
output = Dense(19, activation = "softmax")(pool_rnn) # 等式(6)和(7)
model = Model(inputs = [document, left_context, right_context], outputs = output)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit([X_train_padded_seqs, left_train_padded_seqs, right_train_padded_seqs],y_train,
           batch_size=1024,
           epochs=12,
           validation_data=([X_test_padded_seqs, left_test_padded_seqs, right_test_padded_seqs], y_test))

model.save(model_path + 'textcnn.h5')

# 评价
score = model.evaluate(X_test_padded_seqs, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

## 特征转换
xx_test_word_ids = tokenizer.texts_to_sequences(df_test['word_seg'])
xx_test_padded_seqs = pad_sequences(xx_test_word_ids, maxlen=doc_len)

## 预测
pred_prob = model.predict(xx_test_padded_seqs)
pred = pred_prob.argmax(axis=1)


## 结果保存
df_test['class'] = pred.tolist()
df_test['class'] = df_test['class'] + 1
df_result = df_test.loc[:, ['id','class']]
df_result.to_csv(result_path +'RCNN.csv',index=False)
