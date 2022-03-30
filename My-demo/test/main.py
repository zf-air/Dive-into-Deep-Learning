# -*- coditest_numg: utf-8 -*-

import numpy as np
from sklearn import svm  # 导入svm
from sklearn.externals import joblib # 保存训练模型
from sklearn.decomposition import PCA  # 降维处理
import time # 计算训练时间
import pickle

# step1：读取数据
Train = np.genfromtxt("zip.train.txt")
Test = np.genfromtxt("zip.test.txt")

n = 500
m = 100

train_label = Train[: n, 0]
train_data = Train[: n, 1:]
train_label[train_label != 1] = -1
test_label = Test[: m, 0]
test_data = Test[: m, 1:]
test_label[test_label != 1] = -1
t = time.time()

# step2：进行PCA降维，使运行速度变快

print('star pac')
pca = PCA(n_components=0.9, whiten=True)  # 保留百分之九十的信息
train_x = pca.fit_transform(train_data)
test_x = pca.transform(test_data)
print(train_x.shape)

# step3：svm训练

print('start svm')
svc = svm.SVC(kernel = 'rbf', C = 10) # SVC用于多分类
svc.fit(train_x,train_label)
pre = svc.predict(test_x)

# step4:保存模型

print('save model')
joblib.dump(svc, 'svm_model.m')
joblib.dump(pca, 'pca.m')

# step5:计算准确率

score = svc.score(test_x, test_label)
time = time.time() - t
print(u'准确率：%f,花费时间：%.2fs' % (score, time))


