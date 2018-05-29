#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import operator

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

group, labels = createDataSet()

def classify0(inX,dataSet,labels,k):
    """
    :param inX: 输入要分类的向量数据
    :param dataSet:训练样本集
    :param labels:训练样本集的分类标签
    :param k:要选择的最近邻数目，必须是整数
    :return:前K个标签出现次数统计的元组
    """
    #shape函数返回数据集的维度信息（行,列），shape[0]即数据集的行数
    dataSetSize = dataSet.shape[0]
    #这里要说一下tile()函数，以后我们还会多次用到它
    #在Python2.X中直接使用tile函数，在Python3.X中在np.tile中调用
    # tile(A,B)表示对A重复B次，B可以是int型也可以是数组形式
    # 如果B是int，表示在行方向上重复A，B次，列方向默认为1
    # 如果B是数组形式，tile(A,(B1,B2))表示在行方向上重复B1次，列方向重复B2次
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    #计算diffMat中每一个数据的平方值
    sqDiffMat = diffMat ** 2
    #对每一行数据进行求和，axis=1即按行进行操作
    sqDistances = sqDiffMat.sum(axis=1)
    #求和之后进行开方，计算距离
    distances = sqDistances ** 0.5
    # 排序，这里argsort()返回的是数据从小到大的索引值,这里这就是第几行数据
    sortedDisIndicies = distances.argsort()
    classCount={}
    # 选取距离最小的k个点，并统计每个类别出现的频率
    # 这里用到了字典get(key,default=None)返回键值key对应的值；
    # 如果key没有在字典里，则返回default参数的值，默认为None
    for i in range(k):
        voteIlabel = labels[sortedDisIndicies[i]]
        #这里是选择距离最小的k个点， sortedDistIndicies已经排好序，
        # 只需迭代的取前k个样本点的labels(即标签)，并计算该标签出现的次数，
        # 这里还用到了dict.get(key, default=None)函数，key就是dict中的键voteIlabel，
        # 如果不存在则返回一个0并存入dict，如果存在则读取当前值并+1
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #这里使用了sorted()函数sorted(iterable, cmp=None, key=None, reverse=False)，
    # items()将dict分解为元组列表，operator.itemgetter(1)表示按照第二个元素的次序对元组进行排序，
    #在Python2.X中使用iteritems，在Python3.X中直接使用items（）
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    print(sortedClassCount)
    return sortedClassCount[0][0]

ret = classify0([0,0],group,labels,3)
print(ret)




