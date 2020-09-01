from DatasetGenerator import createDataSet
import math


def calcEntro(dataset):
    sample_count = len(dataset)
    sample_count_per_lable_dict = {}
    for sample in dataset:
        lable = sample[-1]
        if lable not in sample_count_per_lable_dict.keys():
            sample_count_per_lable_dict[lable] = 0
        sample_count_per_lable_dict[lable] += 1

    entro = 0
    for item in sample_count_per_lable_dict.keys():
        prob = sample_count_per_lable_dict.get(item) / sample_count
        entro += -prob * math.log(prob, 2)
    return entro


def splitDataSet(dataSet, axis, value):
    retDataSet = []  # 创建返回的数据集列表
    for featVec in dataSet:  # 遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 去掉axis特征
            reducedFeatVec.extend(featVec[axis + 1:])  
            retDataSet.append(reducedFeatVec)  # 将符合条件的添加到返回的数据集
    return retDataSet  # 返回划分后的数据集


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 特征数量
    baseEntropy = calcEntro(dataSet)  # 计算数据集的香农熵
    bestInfoGain = 0.0  # 信息增益
    bestFeature = -1  # 最优特征的索引值
    for i in range(numFeatures):  # 遍历所有特征
        # 获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 创建set集合{},元素不可重复
        newEntropy = 0.0  # 经验条件熵
        for value in uniqueVals:  # 计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)  # subDataSet划分后的子集
            prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
            newEntropy += prob * calcEntro(subDataSet)  # 根据公式计算经验条件熵
        infoGain = baseEntropy - newEntropy  # 信息增益
        print("第%d个特征的增益为%.3f" % (i, infoGain))  # 打印每个特征的信息增益
        if (infoGain > bestInfoGain):  # 计算信息增益
            bestInfoGain = infoGain  # 更新信息增益，找到最大的信息增益
            bestFeature = i  # 记录信息增益最大的特征的索引值
    return bestFeature


dataset, _ = createDataSet()
# entro = calcEntro(dataset)
beatFeature = chooseBestFeatureToSplit(dataset)
print(beatFeature)