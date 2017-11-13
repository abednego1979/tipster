# -*- coding: utf-8 -*-

#Python 2.7.x

#V0.01
import datetime
import numpy as np

import global_value
from BaseFunc import BaseFunc
from BaseArithmetic import BaseArithmetic

__metaclass__ = type


#多进程
class kNN_Core():
    def __init__(self, device_index=0, kernelFile=''):
        return
    
    def autoNorm(self, dataSet):
        #'''利用cpu或gpu进行数据的归一化'''
        minVals=[]
        maxVals=[]
        ranges=[]
        #获取各个数据的最大最小值
        for i in range(dataSet.shape[1]):
            max_temp, min_temp = self.calcEngine.algorithm_vector_max_min(dataSet[:,i])
            minVals.append(min_temp)
            maxVals.append(max_temp)
        
        minVals=np.array(minVals)
        maxVals=np.array(maxVals)
        ranges=maxVals-minVals
        m=dataSet.shape[0]
        
        #进行矩阵的复制,normDataSet=dataSet.copy()
        normDataSet = self.calcEngine.algorithm_matrix_copy(dataSet)
        
        #矩阵减,normDataSet=normDataSet-np.tile(minVals,(m,1))
        normDataSet = self.calcEngine.algorithm_matrix_vector_sub(normDataSet, minVals)
        
        #矩阵除,normDataSet=normDataSet/np.tile(ranges,(m,1))
        normDataSet = self.calcEngine.algorithm_matrix_vector_div(normDataSet, ranges)
        
        return normDataSet,ranges,minVals    

    
    #dataSet----训练样本,要用第一个数据开始的若干天的数据曲线，与偏移n天的开始的若干天的数据曲线求平均距离
    #DaysLen----dataSet各个列数据，比较曲线的长度。即是比较5天数据曲线，10天数据曲线还是20天数据曲线。
    #labels----标签向量
    #calcLen----要计算的数据量，即第一个数据开始的数据曲线，要与后继偏移多少天内的数据分别求平均距离
    #k----选择最近邻居的个数所占总样本数的比例
    def kNN (self, dataSet, daysLenList, labels, calcLen, kRate):
        #1.计算距离
        dataSetSize=calcLen
        temp_labels=labels[1:]
        
        #求差值
        #diffMat = self.calcEngine.algorithm_matrix_vector_sub(dataSet, inX)
        #求曲线距离差
        diffMat = self.calcEngine.algorithm_matrix_vector_curve_distance(dataSet, daysLenList, calcLen)
        
        #求平方
        sqDiffMat = self.calcEngine.algorithm_matrix_element_square_float(diffMat)
        #每一行求和并对和开方
        distances=self.calcEngine.algorithm_matrix_rowadd_rooting(sqDiffMat)
                
        #下面要对distances和labels进行联合重排
        #把距离量的列表补齐到2的幂次，用0.0补齐
        sortDistIndicies = self.calcEngine.algorithm_argsort(distances)
        
        classCount={}
        
        #选择距离最小的k个点
        voteIlable=0.0      #涨幅
        for i in range(int(kRate*dataSetSize)):
            voteIlable += temp_labels[sortDistIndicies[i]]
        voteIlable /= int(kRate*dataSetSize)#求平均
            
        return voteIlable

    #计算某个权重组合下的预测准确率
    #传入的数据应该是归一化的
    def tipster_kNN(self, calcData, daysLenList, labels, k):
        #将权重值迭加到数据上
        normData = self.calcEngine.algorithm_matrix_mul_k_float(calcData, k)
        
        last_100=[]
        #求最近100天的预测值
        for dayIndex in range(100):
            #从Data[0,:]中取出需要使用的项
            #只用最近的512个点进行预测        
            res=self.kNN(normData, daysLenList, labels, 512, 0.02)#kNN函数返回的是用第1-1001的数据和label预测第0个label的预测值
            last_100.append([res, labels[0]])   #(预测值, 实际值)
            
            normData=normData[1:,:]
            labels=labels[1:]
    
            
        #求当前k系数下的预测准确率
        assert len(last_100)==100
        
        temp_right_rate=np.array([1 if item[0]*item[1]>=0 else 0 for item in last_100]).sum()
        
        
        return [k, temp_right_rate]
    

    
    def weightTrainning(self, ColTitleUsed, weightCoefficient, dbname, db_lock):
        
        assert len(ColTitleUsed)%4 == 0
        
        ColTitleUsed_Name = [item.split('#')[0] for item in ColTitleUsed]   #数据在数据库的列名称
        ColTitleUsed_Days = [int(item.split('#')[1]) for item in ColTitleUsed]   #这个数据要比较N天的曲线之间的距离，这是N的序列
        
        #从数据库中读取全部的列名称
        allDataBaseColTitle=global_value.column_Type_BaseData_Title+global_value.column_Type_ExtendData_Title
        #检查是否有在ColTitleUsed中而不在allColTitle中的列
        t1=[item for item in ColTitleUsed_Name if item not in allDataBaseColTitle]
        assert len(t1)==0,t1
        #从数据库中获取需要的数据，需要是按日期排过序的数据
        #多取的两列，Date是为了排序的需要，PriceIncreaseRate是作为kNN的分类标签
        bData=self.DbEx_GetDataByTitle(dbname, ['Date', 'PriceIncreaseRate']+ColTitleUsed_Name, db_lock, outDataFormat=np.float32)
        bData=bData[1:,:]   #第一条数据是待预测的值，还没有实际的涨跌幅作为kNN的label，所以去除
        
        if bData.shape[0]<640:
            return None        

        #
        validWeightArray=[100.0, 75.0, 50.0, 25.0 ,10.0, 7.5, 5.0, 2.5, 1.0, 0.75, 0.5, 0.25, 0.0]#可以使用的系数选项
        
        #只取1200条数据计算
        dateTime=bData[:640,0]
        labels=bData[:640,1]
        calcData=bData[:640,2:]
        #进行数据归一化
        calcData,ranges,minVals=self.autoNorm(calcData)
        
        assert calcData.dtype==np.float32
        
        #训练过程
        #
        
        trainTurnIndex=0#训练轮次
        lastWeight=weightCoefficient[:] #保存当前的K系数序列，用于与调整后的系数对比，确定哪个更好
        while True:#开启每一轮调整
            
            #对每个k系数遍历
            for k_index,k_title in enumerate(ColTitleUsed):
                #对每个具体k系数，用不同的w替换，得到一组k系数组合
                wc_array=np.array([weightCoefficient]*len(validWeightArray))
                #wc_array=
                #np.array([
                #      [weightCoefficient],
                #      [weightCoefficient],
                #      ...,
                #      [weightCoefficient]])
                
                #将各行可值的第k_index个系数分别替换为validWeightArray中的值
                if k_index+1<len(ColTitleUsed):
                    wc_array=BaseFunc().npColMerge((wc_array[:,0:k_index], np.array(validWeightArray), wc_array[:,k_index+1:]))
                else:
                    wc_array=BaseFunc().npColMerge((wc_array[:,0:k_index], np.array(validWeightArray)))
                    
                #现在wc_array=array([[系数组合1], [系数组合2], ..., [系数组合n(n等于validWeightArray的长度)]])
                
                #现在计算各个K系数组合下的预测准确率
                res=[]
                for k_item in wc_array:
                    res.append(self.tipster_kNN(calcData, ColTitleUsed_Days, labels, k_item))
                
                #这里对各个进程的计算结果进行汇总，得出k_index位置上最优的系数，并调整weightCoefficient
                res.sort(cmp=lambda x,y : cmp(x[1], y[1]), reverse=True)
                best_right_rate=res[0][1]/100.0
                weightCoefficient[k_index]=res[0][0].tolist()[k_index]
                
                if global_value.g_curRunMode==1:
                    #print res
                    print '-'+str(trainTurnIndex)+'-'+str(k_index)+'-'+str(weightCoefficient[k_index])+'-'                
                    print weightCoefficient
                    print best_right_rate
            
            #所有的k系数调整完毕，看看这一轮调整前后系数有没有变化
            if lastWeight==weightCoefficient:
                #如果没有变化，就认为已经得到最优的k系数,这里返回当前的
                return [best_right_rate, lastWeight]
            else:
                lastWeight=weightCoefficient[:]
                    
            trainTurnIndex+=1            
        
        return
            
    def forecast_LastDate(self, ColTitleUsed, weightCoefficient, dbname, db_lock):
        
        assert len(ColTitleUsed)%4 == 0
        
        ColTitleUsed_Name = [item.split('#')[0] for item in ColTitleUsed]   #数据在数据库的列名称
        ColTitleUsed_Days = [int(item.split('#')[1]) for item in ColTitleUsed]   #这个数据要比较N天的曲线之间的距离，这是N的序列        

        #从数据库中读取全部的列名称
        allColTitle=global_value.column_Type_BaseData_Title+global_value.column_Type_ExtendData_Title
        #检查是否有在ColTitleUsed中而不在allColTitle中的列
        t1=[item for item in ColTitleUsed_Name if item not in allColTitle]
        assert len(t1)==0,t1
        #从数据库中获取需要的数据，需要是按日期排过序的数据
        #多取的两列，Date是为了排序的需要，PriceIncreaseRate是作为kNN的分类标签
        bData=self.DbEx_GetDataByTitle(dbname, ['Date', 'PriceIncreaseRate']+ColTitleUsed_Name, db_lock, outDataFormat=np.float32)
        
        if bData.shape[0]<640:
            return None
    
        #只取1200条数据计算
        dateTime=bData[:640,0]
        labels=bData[:640,1]
        calcData=bData[:640,2:]
        #进行数据归一化
        calcData,ranges,minVals=self.autoNorm(calcData)
        
        #将权重值迭加到数据上
        normData = self.calcEngine.algorithm_matrix_mul_k_float(calcData, weightCoefficient)
        
        res=self.kNN(normData, ColTitleUsed_Days, labels, 512, 0.02)#kNN函数返回的是用第1-1001的数据和label预测第0个label的预测值
        return res  #预测涨幅