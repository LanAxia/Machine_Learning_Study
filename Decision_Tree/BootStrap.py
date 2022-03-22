import pandas as pd
import random

from Func import checkPrecision

def bootStrap(dataSet, sampleNum):
    # 拷贝数据，避免更改数据源
    dataSet = dataSet.copy()

    # 创建空的训练集
    trainDataSet = pd.DataFrame(columns=dataSet.columns)

    # 从数据集中随机抽取sampleNum次数据，每一条数据均可被重复抽取，并得到未被抽取的数据
    sampleIndex = {index for index in range(len(dataSet))}
    for i in range(sampleNum):
        # 随机抽取数据
        index = random.randint(0, len(dataSet) - 1)
        trainDataSet = pd.concat([trainDataSet, pd.DataFrame(dataSet.iloc[index, :]).T])

        # 从测试集中去除已被抽取过的数据
        sampleIndex -= {index}

    # 获得测试集完整数据
    testDataSet = dataSet.iloc[list(sampleIndex)]

    return trainDataSet, testDataSet

def bootStrapTest(createTreeFun, classifyFun, dataSet, sampleNum, roundNum=10, labelIndex=-1):
    # 获取特征名和Class列名
    testColumns = dataSet.columns.tolist()
    labelName = testColumns.pop(labelIndex)

    # 计算正确率和混淆矩阵
    accuracy = 0
    matrix = pd.DataFrame(columns=['Pre:1', 'Pre:0'], index=['Real:1', 'Real:0']).fillna(0)

    # 迭代roundNum次
    for i in range(roundNum):
        # 调用bootStrap函数获取训练集和验证集
        trainDataSet, testDataSet = bootStrap(dataSet, sampleNum=sampleNum)

        # 生成决策树模型并预测
        model = createTreeFun(trainDataSet.values.tolist(), trainDataSet.columns.tolist())
        predictionLabels = classifyFun(model, testDataSet[testColumns])['Predict'].tolist()
        realLabels = testDataSet[labelName].tolist()

        # 计算当前的正确率和混淆矩阵
        subAccuracy, subMatrix = checkPrecision(predictionLabels, realLabels)
        accuracy += subAccuracy
        matrix += subMatrix

    # 计算平均正确率和混淆矩阵
    accuracy /= roundNum
    matrix /= roundNum
    matrix = matrix.round(2)

    return accuracy, matrix