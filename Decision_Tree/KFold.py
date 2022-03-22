import pandas as pd
import random

from Func import checkPrecision

# kFold
def kFold(dataSet, curNum, labelIndex=-1, randomShuffle=True):
    # 拷贝数据集，避免修改数据源
    dataSet = dataSet.copy()

    # 计算行数、特征名列表、class的可能取值
    rowNum = len(dataSet)
    labelColumnName = dataSet.columns[labelIndex]
    labels = dataSet[labelColumnName].unique().tolist()

    # 对数据集按照class进行排序，便于按照分布随机划分
    dataSet.sort_values(by=labelColumnName, inplace=True)

    # 创建n折子数据集
    kFoldResult = [[] for i in range(curNum)]

    # 循环获取k折的数据
    for i in range(rowNum // curNum + 1):
        subDataSet = dataSet.iloc[i * curNum : i * curNum + curNum]

        if len(subDataSet) > 0:
            # 打乱当前子数据集
            indexList = subDataSet.index.tolist()
            if randomShuffle:
                random.shuffle(indexList)
                subDataSet = subDataSet.loc[indexList]

            # 为每一折添加至多一行数据
            for i, row in enumerate(subDataSet.values.tolist()):
                kFoldResult[i].append(row)

    # 将n折子数据集转换为DataFrame数据结构
    kFoldResult = [pd.DataFrame(lst, columns=dataSet.columns) for lst in kFoldResult]

    # 迭代产生训练集和验证集
    for i in range(curNum):
        i = curNum - 1 - i

        # 每次挑出一折用于检验，其余用于训练
        trainDataSet = pd.concat(kFoldResult[:i] + kFoldResult[i + 1:], axis=0)
        testDataSet = kFoldResult[i]

        yield trainDataSet, testDataSet

def kFoldTest(modelFun, classifyFun, dataSet, curNum, labelIndex=-1, randomShuffle=True):
    # 获取特征名和Class列名
    testColumns = dataSet.columns.tolist()
    labelName = testColumns.pop(labelIndex)

    # 计正确率和混淆矩阵
    accuracy = 0
    matrix = pd.DataFrame(columns=['Pre:1', 'Pre:0'], index=['Real:1', 'Real:0']).fillna(0)

    # 调用K-Fold循环获得训练数据和验证数据
    for trainDataSet, testDataSet in kFold(dataSet, curNum, randomShuffle=randomShuffle):

        # 构建决策树模型并进行预测
        model = modelFun(trainDataSet.values.tolist(), trainDataSet.columns.tolist())
        predictionLabels = classifyFun(model, testDataSet[testColumns])['Predict'].tolist()
        realLabels = testDataSet[labelName].tolist()

        # 计算当前循环的正确率和混淆矩阵
        subAccuracy, subMatrix = checkPrecision(predictionLabels, realLabels)
        accuracy += subAccuracy
        matrix += subMatrix

    # 计算平均正确率和混淆矩阵
    accuracy /= curNum
    matrix /= curNum
    matrix = matrix.round(2)

    return accuracy, matrix
