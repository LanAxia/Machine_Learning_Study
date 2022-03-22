import random
import pandas as pd

from Func import checkPrecision

# 计算list中的不同元素的个数
def countLst(lst, key=lambda x:x):
    lst = [key(x) for x in lst]
    entities = set(lst)
    result = {entity:0 for entity in entities}
    for x in lst:
        result[x] += 1
    return result

def setAside(dataSet, proportion, labelIndex=-1):
    # 拷贝dataSet取值，创建新的变量，避免更改原dataSet的值
    dataSet = dataSet.copy()

    # 获取类别所在列的列名、类别所有可能取值及其对应的数据子集（字典）
    classLabel = dataSet.columns.tolist()[labelIndex]
    classValues = dataSet[classLabel].unique().tolist()
    classSubDF = {classValue:dataSet[dataSet[classLabel] == classValue] for classValue in classValues}

    # 计算不通class的数据所占的比例
    classTestNums = {classValue:int(len(classSubDF[classValue]) * proportion) for classValue in classValues}

    # 分离训练集和验证集
    trainDataSet = pd.DataFrame(columns=dataSet.columns)
    testDataSet = pd.DataFrame(columns=dataSet.columns)
    for classValue in classValues:
        classDataSet = classSubDF[classValue]

        # 打乱同一类别的数据
        classDataSetIndex = classDataSet.index.tolist()
        random.shuffle(classDataSetIndex)
        classDataSet = classDataSet.loc[classDataSetIndex]

        # 将打乱后的数据按照比例切割并存入训练集和验证集
        testClassDataSet = classDataSet.iloc[:classTestNums[classValue]]
        trainClassDataSet = classDataSet.iloc[classTestNums[classValue]:]
        testDataSet = pd.concat([testDataSet, testClassDataSet])
        trainDataSet = pd.concat([trainDataSet, trainClassDataSet])

    return trainDataSet, testDataSet

def setAsideTest(createTreeFun, classifyFun, dataSet, roundNum=10, proportion=0.2, labelIndex=-1):
    # 获取所有特征名的列表，并删除Class所在的列
    testColumns = dataSet.columns.tolist()
    labelName = testColumns.pop(labelIndex)

    # 计算正确率和混淆矩阵
    accuracy = 0
    matrix = pd.DataFrame(columns=['Pre:1', 'Pre:0'], index=['Real:1', 'Real:0']).fillna(0)

    # 多次计算取正确率和混淆矩阵的平均值
    for i in range(roundNum):
        # 拆分训练集和验证集
        trainDataSet, testDataSet = setAside(dataSet, proportion=proportion, labelIndex=labelIndex)

        # 创建分类模型
        model = createTreeFun(trainDataSet.values.tolist(), trainDataSet.columns.tolist())
        predictionLabels = classifyFun(model, testDataSet[testColumns])['Predict'].tolist()
        realLabels = testDataSet[labelName].tolist()

        # 计算准确率
        subAccuracy, subMatrix = checkPrecision(predictionLabels, realLabels)
        accuracy += subAccuracy
        matrix += subMatrix

    # 计算平均正确率和混淆矩阵
    accuracy /= roundNum
    matrix /= roundNum
    matrix = matrix.round(2)

    return accuracy, matrix