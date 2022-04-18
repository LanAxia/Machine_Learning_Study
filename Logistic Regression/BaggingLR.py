import multiprocessing
from multiprocessing import Process
import pandas as pd
import numpy as np
import pickle
import math
import random
import time

class BaggingLR(object):
    def __init__(self, bagCount, attNum):
        self.bagIDs = set(range(bagCount))
        self.estmGenerators = dict()
        self.columns = tuple()
        self.estimators = dict()
        self.predictResults = dict()
        self.bagColumns = dict()
        self.bagRows = dict()
        self.shareDict = multiprocessing.Manager().dict()
        self.shareDict['estimators'] = dict()
        self.shareDict['predictResults'] = dict()
        self.bagCount = bagCount
        self.attNum = attNum
    
    @staticmethod
    def checkGradThreshold(grad, threshold):
        '''
        当有一个元素比阈值大，就返回True
        '''
        check = np.abs(grad) - threshold
        if True in (check > 0).ravel():
            return True
        else:
            return False

    @staticmethod
    def sigmoid(inX):
        x_shape = inX.shape                                      # 获取输入矩阵形状
        x = np.array(inX).ravel()                                # 铺平
        y = []
        for i in range(len(x)):
            if  x[i] >= 0:
                y.append(1 / (1 + np.exp(-x[i])))
            else:
                y.append(np.exp(x[i]) / (1 + np.exp(x[i])))      # 当某一个元素小于0时，用另一个公式计算，解决上溢问题
        return np.matrix(np.array(y).reshape(inX.shape))
    
    @staticmethod
    def getSampleIndex(sampleNum, size):
        return [random.randint(0, size - 1) for i in range(sampleNum)]

    @staticmethod
    def miniBatchGradAscent(dataMatIn, classLabels, batchSize=10, gradThreshold=1e-10, jThreshold=0.95, maxCycles=2000):
        # 初始化参数
        m, n = dataMatIn.shape
        dataMatrix = np.matrix(dataMatIn)
        labelMat = np.matrix(np.array(classLabels).reshape((m, 1)))
        weights = np.matrix(np.ones((n, 1)))
        dataIndex = list(range(len(dataMatrix)))
        
        # 最多循环maxCycles次
        for i in range(maxCycles):
            dataIndex = list(range(m))
            random.shuffle(dataIndex)
            errorRate = 0
            # 获取随机小批量数据
            for j in range(math.floor(m / batchSize) + 1):
                randomIndex = dataIndex[batchSize * j:batchSize * j + batchSize]       # b，从数据中抽取batchSize行用于更新数据
                if len(randomIndex) > 0:
                    alpha = 4 / (1.0 + i + j) + 0.01                                   # 随着轮数的增加，更新的步幅变小
                    batchData = dataMatrix[randomIndex]                                # b * n
                    batchLabels = labelMat[randomIndex]                                # b * 1

                    # 用小批量数据更新weights
                    h = BaggingLR.sigmoid(batchData * weights)                                   # b * 1
                    error = batchLabels - h                                            # b * 1
                    errorRate += np.abs(error).sum()
                    gradient = batchData.transpose() * error                           # n * 1
                    if not BaggingLR.checkGradThreshold(gradient, gradThreshold):
                        return weights
                    weights += alpha * gradient
            
            if 1 - errorRate / m > jThreshold:
                return weights
        return weights

    @staticmethod
    def classifyMatrix(inX, weights):
        inX = np.matrix(inX)
        weights = np.matrix(weights)
        probs = BaggingLR.sigmoid(inX * weights)
        return np.round(probs)
    
    def generateEstimator(self, bagID, para):
        weights = BaggingLR.miniBatchGradAscent(**para)
        est = self.shareDict['estimators']
        est[bagID] = weights
        self.shareDict['estimators'] = est

    def fit(self, dataMatIn, classLabels, **lrparameters):
        self.__init__(self.bagCount, self.attNum) # 初始化对象
        self.columns = tuple(dataMatIn.columns.tolist())
        for id in self.bagIDs:
            # 生成子袋所拟合的列
            columnNum = abs(math.floor(self.attNum)) + 1
            newColumn = list(self.columns)
            random.shuffle(newColumn)
            columns = newColumn[:columnNum]
            self.bagColumns[id] = columns
            self.bagRows[id] = BaggingLR.getSampleIndex(len(dataMatIn), len(dataMatIn))
            bagParameters = {
                    'dataMatIn':dataMatIn[columns].copy(), 
                    'classLabels':classLabels.copy(), 
                }
            for key in lrparameters:
                bagParameters[key] = lrparameters[key]
            
            self.estmGenerators[id] = bagParameters
        
        processes = []
        for bagID in self.estmGenerators:
            para = self.estmGenerators[bagID]
            process = Process(target=self.generateEstimator, args=(bagID, para))
            processes.append(process)
        
        for process in processes:
            process.start()
        
        for process in processes:
            process.join()
        
        self.estimators = self.shareDict['estimators']
    
    def predictEstm(self, bagID, dataMatIn):
        weights = self.estimators[bagID]
        columns = self.bagColumns[bagID]
        subSet = dataMatIn[columns]
        result = BaggingLR.classifyMatrix(subSet.values, weights)
        prt = self.shareDict['predictResults']
        prt[bagID] = result
        self.shareDict['predictResults'] = prt

    def predict(self, dataMatIn):
        processes = []
        for bagID in self.bagIDs:
            p = Process(target=self.predictEstm, args=(bagID, dataMatIn))
            processes.append(p)
        
        for process in processes:
            process.start()
        
        for process in processes:
            process.join()
        
        self.predictResults = self.shareDict['predictResults']

        results = [self.predictResults[bagID] for bagID in self.predictResults]
        resultMatrix = np.concatenate(results, axis=1).sum(axis=1) / self.bagCount
        resultMatrix = np.round(resultMatrix)
        return np.array(resultMatrix).reshape((1, -1))[0]

def calPrecRate(predResult, origResult):
    predResult = np.matrix(predResult)
    origResult = np.matrix(origResult)
    m, _n = predResult.shape

    return float(sum(predResult == origResult)) / m

def calConfMatrix(predResult, origResult):
    predResult = np.matrix(predResult)
    origResult = np.matrix(origResult)
    m, _n = predResult.shape

    # 通过计算准确率的方式计算混淆矩阵的四个值，最终返回混淆矩阵（pd.DataFrame）和正确率
    zeroZero = calPrecRate(predResult + origResult, np.zeros((m, 1))) * m        # 0 + 0 = 0
    zeroOne = calPrecRate(predResult - origResult, np.ones((m, 1))) * m          # 1 - 0 = 0
    oneZero = calPrecRate(origResult - predResult, np.ones((m, 1))) * m          # 1 - 0 = 0
    oneOne = calPrecRate(predResult + origResult, 2 * np.ones((m, 1))) * m       # 1 + 1 = 2
    return pd.DataFrame([[zeroZero, zeroOne], [oneZero, oneOne]], columns=['zero', 'one'], index=['zero', 'one']), (zeroZero + oneOne) / (zeroZero + oneOne + zeroOne + oneZero)

if __name__ == '__main__':
    with open('./Code/Cache/data_outcome.pkl', 'rb') as f:
        trainDataSet, trainLabels, testDataSet, testLabels, columns = pickle.load(f)
        trainDataSet = pd.DataFrame(trainDataSet, columns=columns)
        testDataSet = pd.DataFrame(testDataSet, columns=columns)
        trainLabels = pd.DataFrame(trainLabels, columns=['Labels'])
        testLabels = pd.DataFrame(testLabels, columns=['Labels'])

    bagNum = 10
    attributeNum = 13
    baggingLR = BaggingLR(bagNum, attributeNum)
    startTime = time.time()
    baggingLR.fit(trainDataSet, trainLabels.values.T, jThreshold=0.95, batchSize=50)
    predictResult = baggingLR.predict(testDataSet)
    endTime = time.time()
    confMatrix, accuracy = calConfMatrix(predictResult.reshape((-1, 1)), testLabels.values)
    print('通过random算法获取{}个子logistic regression分类器，每个分类器对{}个属性进行运算\n使用改进的小批量梯度上升进行Logistic Regression分类\n通过投票方式获取最终分类结果\n精确率有{:.03f}%\n共计耗时{:.03f}秒'.format(bagNum, attributeNum, accuracy * 100, endTime -  startTime))
