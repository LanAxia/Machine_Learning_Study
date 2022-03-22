from math import log2

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

# 根据给定的feature分离数据集
def splitData(dataSet, axis, value):
    # 遍历每一行数据，当该行数据的特征取值等于给定的特征取值时，将其返回
    retDataSet = []

    for featVec in dataSet:
        if featVec[axis] == value:
            featVec = featVec[:axis] + featVec[axis + 1:]     # 通过相加重新创建一个值，避免影响原有数据
            retDataSet.append(featVec)
    return retDataSet

# 找到信息熵增益最大的feature
def chooseBestFeatureToSplit(dataSet):
    # 计算一些用于辨识最大信息增益的变量
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain, bestFeature = 0, -1         # 如果没有合理的信息熵增益则会返回-1，需要在createTree函数中进行处理

    # 遍历所有特征
    for i in range(numFeatures):
        # 计算该特征的所有可能的取值
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)

        # 对每一种可能的取值计算信息熵
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitData(dataSet, i, value)
            prob = len(subDataSet) / len(dataSet)
            newEntropy += prob * calcShannonEnt(subDataSet)

        # 计算信息增益
        infoGain = baseEntropy - newEntropy

        # 获得最大的信息增益
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature

# 找到classList中出现次数最多的元素
def majorityCnt(classList):
    # 调用countLst计算class列中每个class出现的次数
    classCount = countLst(classList)

    # 返回出现次数最多的class
    return max(classCount, key=lambda x:classCount[x])

# 创建ID3决策树
def createID3Tree(dataSet, labels, featureValues=None):
    # 创建节点
    myTree = dict()
    dataSet = dataSet.copy()                          # 避免修改原数据
    classList = [row[-1] for row in dataSet]          # 得到class列

    # 如果类别完全相同则停止继续分类
    if len(set(classList)) == 1:
        return classList[0]

    # 如果特征集为空，返回占比最高的class
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 若所有样本在所有属性集上取值相同，返回占比最高的class
    if len(set([tuple(row[:-1]) for row in dataSet])) == 1:
        return majorityCnt(classList)

    # 创建一个featureValues的字典用于指导构建决策树的分枝，该程序块只有在第一次递归中运行，之后会反复调用featureValues
    if featureValues == None:
        featureValues = dict()
        for i, feature in enumerate(labels):
            featureValues[feature] = list(set([x[i] for x in dataSet]))

    # 找到最优的划分属性及其label
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 若bestFeat = -1则表示任何划分都不会带来信息增益，此时终止递归并返回占比最高的class
    if bestFeat == -1:
        return majorityCnt(classList)
    bestFeatLabel = labels[bestFeat]

    # 在labels中去除最优的属性
    myTree[bestFeatLabel] = dict()
    del(labels[bestFeat])

    # 获取最优的属性的列以及该属性所有的取值
    featValues = [row[bestFeat] for row in dataSet]
    uniqueValues = featureValues[bestFeatLabel]

    '''
    构建子树：
    与书上代码不同的是，这里引入了featureValues变量记录每个feature所有可能的取值。
    在构建子树时，对feature的所有可能取值构建子树，若当前数据集已没有部分取值，则子树为当前数据集中占比最高的class。
    这样做避免部分子树数据集少导致后续分类时，无法对部分测试集作出分类，提高决策树的泛化能力。
    '''
    for value in uniqueValues:
        subLabels = labels.copy()                                   # 避免在多次递归中反复对某一label列表进行修改
        subData = splitData(dataSet, bestFeat, value)

        # 递归构建决策树
        if len(subData) == 0:                                       # 若子数据集为空，则返回当前dataSet中的占比最高的class
            myTree[bestFeatLabel][value] = majorityCnt(classList)
        else:
            myTree[bestFeatLabel][value] = createID3Tree(subData, subLabels, featureValues)
    return myTree

# 根据决策树模型对某一条记录进行预测
def classifyID3(inputTree, featLabels, testVec):
    # 计算相关变量
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    classLabel = None

    # 根据测试向量迭代查询决策树
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            next_dict = secondDict[key]

            # 若子树为树状结构，则迭代查询，若已至叶子结点，则返回class
            if isinstance(next_dict, dict):
                classLabel = classifyID3(next_dict, featLabels, testVec)
            else:
                classLabel = next_dict
    return classLabel

# 根据决策树模型对多条记录进行预测
def classifyID3DF(inputTree, testDF):
    # 获取featLabels参数，即获得所有预测需要使用的label
    testDF = testDF.copy()
    featLabels = testDF.columns.tolist()

    # 对每一条测试数据进行预测，并将预测结果存储在Predict列中
    testVecs = testDF.values.tolist()
    result = [classifyID3(inputTree, featLabels, testVec) for testVec in testVecs]
    testDF['Predict'] = result

    return testDF

# Improved ID3 Tree
# 创建ID3决策树
def createID3TreeImproved(dataSet, labels, featureValues=None):
    # 创建节点
    myTree = dict()
    dataSet = dataSet.copy()                          # 避免修改原数据
    classList = [row[-1] for row in dataSet]          # 得到class列

    # 如果类别完全相同则停止继续分类
    if len(set(classList)) == 1:
        return classList[0]

    # 如果特征集为空，返回占比最高的class
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 若所有样本在所有属性集上取值相同，返回占比最高的class
    if len(set([tuple(row[:-1]) for row in dataSet])) == 1:
        return majorityCnt(classList)

    # 创建一个featureValues的字典用于指导构建决策树的分枝，该程序块只有在第一次递归中运行，之后会反复调用featureValues
    if featureValues == None:
        featureValues = dict()
        for i, feature in enumerate(labels):
            featureValues[feature] = list(set([x[i] for x in dataSet]))

    # 找到最优的划分属性及其label
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 若bestFeat = -1则表示任何划分都不会带来信息增益，此时终止递归并返回占比最高的class
    if bestFeat == -1:
        return majorityCnt(classList)
    bestFeatLabel = labels[bestFeat]

    # 在labels中去除最优的属性
    myTree[bestFeatLabel] = dict()
    del(labels[bestFeat])

    # 获取最优的属性的列以及该属性所有的取值
    featValues = [row[bestFeat] for row in dataSet]
    uniqueValues = featureValues[bestFeatLabel]

    '''
    构建子树：
    与普通的ID3树不同得是，在这里每一层加入了major字段。
    当分类时出现训练时没有出现过的属性取值或缺少某个属性时，返回当前情况下最可能的分类。
    提高模型的鲁棒性和泛化性。
    '''
    myTree[bestFeatLabel]['major'] = majorityCnt(classList)
    for value in uniqueValues:
        subLabels = labels.copy()                                   # 避免在多次递归中反复对某一label列表进行修改
        subData = splitData(dataSet, bestFeat, value)

        # 递归构建决策树
        if len(subData) == 0:                                       # 若子数据集为空，则返回当前dataSet中的占比最高的class
            myTree[bestFeatLabel][value] = majorityCnt(classList)
        else:
            myTree[bestFeatLabel][value] = createID3TreeImproved(subData, subLabels, featureValues)
    return myTree

# 根据决策树模型对某一条记录进行预测
def classifyID3Improved(inputTree, featLabels, testVec):
    # 计算相关变量
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
                classLabel = classifyID3Improved(next_dict, featLabels, testVec)
            else:
                classLabel = next_dict
        else:
            classLabel = secondDict['major']
    else:
        classLabel = secondDict['major']
    return classLabel

# 根据决策树模型对多条记录进行预测
def classifyID3ImprovedDF(inputTree, testDF):
    # 获取featLabels参数，即获得所有预测需要使用的label
    testDF = testDF.copy()
    featLabels = testDF.columns.tolist()

    # 对每一条测试数据进行预测，并将预测结果存储在Predict类中
    testVecs = testDF.values.tolist()
    result = [classifyID3Improved(inputTree, featLabels, testVec) for testVec in testVecs]
    testDF['Predict'] = result

    return testDF