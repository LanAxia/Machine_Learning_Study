from math import log2

# 计算list中的不同元素的个数
def countLst(lst, key=lambda x:x):
    lst = [key(x) for x in lst]
    entities = set(lst)
    result = {entity:0 for entity in entities}
    for x in lst:
        result[x] += 1
    return result

# 计算信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)

    # 计算每一种label的数量
    labelCounts = countLst(dataSet, key=lambda x:x[-1])

    # 计算信息熵
    shannonEnt = 0
    for key in labelCounts.keys():
        prob = labelCounts[key] / numEntries
        shannonEnt -= prob * log2(prob)
    return shannonEnt

# 计算固有值IV
def calcIV(featList):
    numEntries = len(featList)
    numValueCount = countLst(featList)

    # 计算IV
    iv = 0
    for value in numValueCount:
        pValue = numValueCount[value] / numEntries
        iv -= (pValue) * log2(pValue)

    return iv

# 根据给定的feature分离数据集
def splitData(dataSet, axis, value):
    # 遍历每一行数据，当该行数据的特征取值等于给定的特征取值时，将其返回
    retDataSet = []

    for featVec in dataSet:
        if featVec[axis] == value:
            featVec = featVec[:axis] + featVec[axis + 1:]
            retDataSet.append(featVec)
    return retDataSet

# 找到信息增益率最大的feature
def chooseBestFeatureToSplit(dataSet):
    # 计算一些用于辨识最大信息增益率的变量
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGainRatio, bestFeature = 0, -1

    # 遍历所有特征
    for i in range(numFeatures):
        # 计算该特征的所有可能的取值
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)

        # 对每一种可能的取值计算信息增益
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitData(dataSet, i, value)
            prob = len(subDataSet) / len(dataSet)
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy

        # 计算信息增益率
        iv = calcIV(featList)
        if iv == 0:
            continue  # 当IV=0时，即该属性只有一种取值对这一属性进行分类没有意义
        infoGainRatio = infoGain / iv

        if infoGainRatio > bestInfoGainRatio:
            bestInfoGainRatio = infoGainRatio
            bestFeature = i

    return bestFeature

# 找到classList中出现次数最多的元素
def majorityCnt(classList):
    # 调用countLst计算class列中每个class出现的次数
    classCount = countLst(classList)

    # 返回出现次数最多的class
    return max(classCount, key=lambda x:classCount[x])

def createC45Tree(dataSet, labels, featureValues=None):
    # 创建节点
    myTree = dict()
    dataSet = dataSet.copy()
    classList = [example[-1] for example in dataSet]

    # 如果类别完全相同则停止继续分类
    if len(set(classList)) == 1:
        return classList[0]

    # 如果特征集为空，返回占比最高的class
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 若所有样本在所有属性集上取值相同，返回占比最高的class
    if len(set([tuple(row[:-1]) for row in dataSet])) == 1:
        return majorityCnt(classList)

    # 创建一个featureValues的字典用于指导构建决策树的分枝
    if featureValues == None:
        featureValues = dict()
        for i, feature in enumerate(labels):
            featureValues[feature] = list(set([x[i] for x in dataSet]))

    # 找到最优的划分属性方式
    bestFeat = chooseBestFeatureToSplit(dataSet)
    if bestFeat == -1:                                                                 # # 如果信息增益率为负则会返回bestFeat为-1，此时也提前结束分类
        return majorityCnt(classList)
    bestFeatLabel = labels[bestFeat]

    # 在labels中去除最优的属性
    myTree[bestFeatLabel] = dict()
    del(labels[bestFeat])

    # 获取最优的属性的列以及该属性所有的取值
    featValues = [row[bestFeat] for row in dataSet]
    uniqueValues = featureValues[bestFeatLabel]

    # 构建子树
    myTree[bestFeatLabel]['major'] = majorityCnt(classList)
    for value in uniqueValues:
        subLabels = labels.copy()                                   # 避免在多次递归中反复对某一label列表进行修改
        subData = splitData(dataSet, bestFeat, value)

        # 递归构建决策树
        if len(subData) == 0:                                       # 若子数据集为空，则返回当前dataSet中的占比最高的class
            myTree[bestFeatLabel][value] = majorityCnt(classList)
        else:
            myTree[bestFeatLabel][value] = createC45Tree(subData, subLabels, featureValues)
    return myTree

# 用C4.5树模型进行分类
def classifyC45(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # 当特征名不存在于featLabels，即测试集不包含该属性时返回当前情况下最有可能的class
    if firstStr in featLabels:
        featIndex = featLabels.index(firstStr)

        # 根据测试向量迭代查询决策树，当该属性的取值不存在于决策树的子树上时则返回当前情况下最有可能的class
        featValue = testVec[featIndex]
        if featValue in list(secondDict.keys()):
            next_dict = secondDict[featValue]

            # 若子树为树状结构，则迭代查询，若已至叶子结点，则返回class
            if isinstance(next_dict, dict):
                classLabel = classifyC45(next_dict, featLabels, testVec)
            else:
                classLabel = next_dict
        else:
            classLabel = secondDict['major']
    else:
        classLabel = secondDict['major']
    return classLabel

# 封装可以直接传入DataFrame的分类函数，将分类结果直接合并进DataFrame中
def classifyC45DF(inputTree, testDF):
    # 获取featLabels参数，即获得所有预测需要使用的label
    testDF = testDF.copy()
    featLabels = testDF.columns.tolist()

    # 对每一条测试数据进行预测，并将预测结果存储在Predict类中
    testVecs = testDF.values.tolist()
    result = [classifyC45(inputTree, featLabels, testVec) for testVec in testVecs]
    testDF['Predict'] = result

    return testDF