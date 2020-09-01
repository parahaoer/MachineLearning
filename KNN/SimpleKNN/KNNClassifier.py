import numpy as np
import operator

class KNNClassifier():
    def createDataSet(self):
        # 四组二维特征
        group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
        # 四组特征的标签
        labels = ['爱情片', '爱情片', '动作片', '动作片']
        return group, labels

    def classifier(self, testData, trainset, labels, k):
        
        distance = np.sum((testData - trainset) ** 2) ** 0.5
         
        # 对距离从小到大排序，并返回索引值列表
        sorted_distance_indies = distance.argsort()

        # 取出
        classCount = {}
        for i in range(k):

            label = labels[sorted_distance_indies[i]]
            classCount[label] = classCount.get(label, 0) + 1
        # 根据classCount的value 降序排序，  
        sorted_classCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_classCount[0][0]


if __name__ == "__main__":
    kNNClassifier = KNNClassifier()
    trainset, labels = kNNClassifier.createDataSet()

    testdata = [2, 10]

    res = kNNClassifier.classifier(testdata, trainset, labels, 3)

    print(res)
