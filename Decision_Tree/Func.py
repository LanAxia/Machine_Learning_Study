import pandas as pd

# 计算预测的准确率
def checkPrecision(predictLst, realLst):
    accuracy = 0
    TP, TN, FP, FN = 0, 0, 0, 0

    # 遍历所有预测结果，计算TP、TN、FN、FP
    for x, y in zip(predictLst, realLst):
        if x == y:
            if x == 1:
                TP += 1
            else:
                TN += 1
            accuracy += 1
        else:
            if x == 1:
                FN += 1
            else:
                FP += 1

    TP *= 100 / len(predictLst)
    TN *= 100 / len(predictLst)
    FP *= 100 / len(predictLst)
    FN *= 100 / len(predictLst)

    # 计算准确率
    accuracy /= len(predictLst)

    # 输出混淆矩阵
    matrix = pd.DataFrame([[TP, FP], [FN, TN]], columns=['Pre:1', 'Pre:0'],\
                          index=['Real:1', 'Real:0'])

    return accuracy, matrix