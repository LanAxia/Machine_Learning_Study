import re
import pandas as pd
from math import log2

from Func import checkPrecision

# 正式的Cart树
# 计算list中的不同元素的个数
def countLst(lst, key=lambda x:x):
    # 可以通过key参数对复杂的可迭代对象计算其元素出现的个数

    # 获取List中可能存在的全部取值
    lst = [key(x) for x in lst]
    entities = set(lst)

    # 用字典形式计算个数
    result = {entity:0 for entity in entities}
    for x in lst:
        result[x] += 1
    return result

# 计算Gini指数
def calcGini(dataSet):
    # 获取class列及其所有可能的取值
    classList = [row[-1] for row in dataSet]
    numEntries = len(classList)
    labelCounts = countLst(classList)

    # 计算Gini指数
    gini = 1
    for label in list(labelCounts.keys()):
        prob = labelCounts[label] / numEntries
        gini -= (prob ** 2)

    return gini

'''
根据给定的feature分离数据集。
与之前的方法不同的是，该分离函数会同时返回符合该特征取值的子数据集和不符合该特征取值的子数据集。
对于符合该值的子集，为方便后续计算，会删除该特征列；而对于不符合的子集，则不会删除。
'''
def splitData(dataSet, axis, value):
    # 分别存储符合特征取值的子集和不符合的子集
    dataSet = dataSet.copy()
    retDataSet = []
    otherDataSet = []

    # 遍历每一行数据，分别存储数据
    for featVec in dataSet:
        if featVec[axis] == value:
            featVec = featVec[:axis] + featVec[axis + 1:]
            retDataSet.append(featVec)
        else:
            otherDataSet.append(featVec)
    return retDataSet, otherDataSet

# 找到最适合切分的feature和value
def chooseBestValueToSplit(dataSet):
    bestFeature = None
    bestValue = None
    bestGini = 1

    # 遍历取得gini最小的Feature和Value，确定最优划分
    for feature in range(len(dataSet[0]) - 1):
        featureList = [row[feature] for row in dataSet]
        for value in set(featureList):
            retDataSet, otherDataSet = splitData(dataSet, feature, value)
            retDataSetGini = calcGini(retDataSet)
            otherDataSetGini = calcGini(otherDataSet)
            try:
                gini = ((len(retDataSet) / len(dataSet)) * retDataSetGini + (len(otherDataSet) / len(dataSet)) * otherDataSetGini)
                giniComplex = gini * len(dataSet) / (len(retDataSet) * len(otherDataSet) + 1)
            except Exception as e:
                print(e)
            if giniComplex <= bestGini:
                bestFeature = feature
                bestValue = value
                bestGini = giniComplex

    return bestFeature, bestValue

# 找到classList中出现次数最多的元素
def majorityCnt(classList):
    # 调用countLst计算class列中每个class出现的次数
    classCount = countLst(classList)

    # 返回出现次数最多的class
    return max(classCount, key=lambda x:classCount[x])

# 创建cart决策树
def createCartTree(dataSet, testDataSet, labels, numThreshold=1, giniThreshold=0.2):
    # 创建叶子节点
    cartTree = dict()
    dataSet = dataSet.copy()
    classList = [row[-1] for row in dataSet]

    # 当节点中的样本个数小于预定阈值，结束递归
    if len(dataSet) < numThreshold:
        testValueCounts = countLst([row[-1] for row in testDataSet])
        majorClass = majorityCnt(classList)
        if majorClass in testValueCounts:
            accuracyNow = testValueCounts[majorClass] / len(testDataSet)
        else:
            accuracyNow = 0
        return majorityCnt(classList), accuracyNow

    # 当没有更多特征，结束递归
    elif len(dataSet[0]) == 1:
        testValueCounts = countLst([row[-1] for row in testDataSet])
        majorClass = majorityCnt(classList)
        if majorClass in testValueCounts:
            accuracyNow = testValueCounts[majorClass] / len(testDataSet)
        else:
            accuracyNow = 0
        return majorityCnt(classList), accuracyNow

    # 当数据集的基尼指数小于预定阈值时，结束递归
    elif calcGini(dataSet) < giniThreshold:
        testValueCounts = countLst([row[-1] for row in testDataSet])
        majorClass = majorityCnt(classList)
        if majorClass in testValueCounts:
            accuracyNow = testValueCounts[majorClass] / len(testDataSet)
        else:
            accuracyNow = 0
        return majorityCnt(classList), accuracyNow

    elif len(testDataSet) == 0:
        return majorityCnt(classList), 1

    else:
        # 找到最优划分的Feature和Value
        bestFeature, bestValue = chooseBestValueToSplit(dataSet)
        # 当不存在最优划分时，返回当前最可能的class
        if bestValue is None:
            testValueCounts = countLst([row[-1] for row in testDataSet])
            majorClass = majorityCnt(classList)
            if majorClass in testValueCounts:
                accuracyNow = testValueCounts[majorClass] / len(testDataSet)
            else:
                accuracyNow = 0
            return majorityCnt(classList), accuracyNow

        # 获取最优划分Feature的Label
        bestFeatureLabel = labels[bestFeature]
        # 获取最优划分Feature列
        featList = [row[bestFeature] for row in dataSet]


        # 根据最优划分拆分数据集
        trueSubDataSet, falseSubDataSet = splitData(dataSet, bestFeature, bestValue)
        # 当匹配为True时，不需要再匹配该特征，因此删除该列
        trueLabels = labels[:bestFeature] + labels[bestFeature + 1:]
        # 当匹配为False时，可能还需要再匹配该特征，因此不删除该列
        falseLabels = labels.copy()

        # 根据最优划分拆分验证集
        trueSubTestDataSet, falseSubTestDataSet = splitData(testDataSet, bestFeature, bestValue)

        # 计算当前的准确率
        majorClassNow = majorityCnt(classList)
        testValueCounts = countLst([row[-1] for row in testDataSet])
        if majorClassNow not in testValueCounts:
            testValueCounts[majorClassNow] = 0
        try:
            accuracyNow = testValueCounts[majorClassNow] / len(testDataSet)
        except Exception as e:
            print(e)

        # 递归构建子树
        cartTree['%s_%s'%(bestFeatureLabel, str(bestValue))] = {True : None, False : None}
        trueTree, trueAccuracy = createCartTree(trueSubDataSet, trueSubTestDataSet, trueLabels, numThreshold=numThreshold, giniThreshold=giniThreshold)

        # 当匹配为False的子树没有数据时，返回当前（在划分True和False子集之前的数据集中）最有可能的class，否则递归构建子树
        if len(falseSubDataSet) == 0:
            falseTree = majorityCnt(classList)
            falseValueCounts = countLst([row[-1] for row in falseSubTestDataSet])
            if falseTree in list(falseValueCounts.keys()):
                falseAccuracy = falseValueCounts[falseTree] / len(falseSubTestDataSet)
            else:
                falseAccuracy = 0
        else:
            falseTree, falseAccuracy = createCartTree(falseSubDataSet, falseSubTestDataSet, falseLabels, numThreshold=numThreshold, giniThreshold=giniThreshold)

        subAccuracy = (trueAccuracy * len(trueSubTestDataSet) + falseAccuracy * len(falseSubTestDataSet)) / len(testDataSet)
        if subAccuracy > accuracyNow:
            cartTree['%s_%s'%(bestFeatureLabel, str(bestValue))][False] = falseTree
            cartTree['%s_%s'%(bestFeatureLabel, str(bestValue))][True] = trueTree
        else:
            cartTree = majorityCnt(classList)
            print(accuracyNow)
            print(subAccuracy)

        return cartTree, accuracyNow

# 根据Cart树对测试集进行分类
ptn = re.compile('(\w+)_(-?\d+)')
def classifyCart(inputTree, labels, testVec):
    # 根据当前的"特征_取值"使用正则表达式获取特征和取值
    featureValue = ptn.search(list(inputTree.keys())[0])
    if featureValue:
        featureLabel = featureValue.group(1)
        feature = labels.index(featureLabel)
        value = int(featureValue.group(2))

        # 获取下一层子树，若为树状结构则递归查询，否则直接返回该值
        secondDict = inputTree[featureValue.group(0)]
        if testVec[feature] == value:
            nextDict = secondDict[True]
            if isinstance(nextDict, dict):
                labels = labels[:feature] + labels[feature + 1:]
                testVec = testVec[:feature] + testVec[feature + 1:]
                return classifyCart(nextDict, labels, testVec)
            else:
                return nextDict
        else:
            nextDict = secondDict[False]
            if isinstance(nextDict, dict):
                return classifyCart(nextDict, labels, testVec)
            else:
                return nextDict

# 根据Cart树对多条测试集进行预测
def classifyCartDF(inputTree, dataSet):
    # 获取featLabels参数，即获得所有预测需要使用的label
    dataSet = dataSet.copy()
    features = dataSet.columns.tolist()

    # 对每一条测试数据进行预测，并将预测结果存储在Predict列中
    predict = []
    for testVec in dataSet.values.tolist():
        result = classifyCart(inputTree, features, testVec)
        predict.append(result)
    dataSet['Predict'] = predict
    return dataSet