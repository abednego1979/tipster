# -*- coding: utf-8 -*-

#Python 2.7.x

#V0.01
import requests
import os
import traceback
import warnings
import json
import base64
import numpy as np

import global_value
from DB_Ex import MyDbEx
from BaseArithmetic import BaseArithmetic
from BaseVariable import BaseVariable
from kNN02 import kNN_Core

__metaclass__ = type


class InitSockData(BaseVariable, MyDbEx, BaseArithmetic, kNN_Core):
    calcEngine=None
    
    def __init__(self, calcEngine=None):
        self.calcEngine=calcEngine
        BaseVariable.__init__(self)
        MyDbEx.__init__(self)
        BaseArithmetic.__init__(self)
        kNN_Core.__init__(self)
        return
    
    #获取所有A股股票----长周期任务
    #这是一次性的工作，方法是1.从http://www.sse.com.cn/assortment/stock/list/share/和http://www.szse.cn/main/marketdata/jypz/colist/获取
    #两市的xls格式的列表，2.合并列表，保留“A股代码/A股简称/A股上市日期/A股总股本/A股流通股本”这5个基本信息，3.筛选去除无效的一些数据， 4.去掉表头，5.将文件修改为csv格式
    #注意两市在表示时间的格式上有所不同
    
    
    #从我们主动整理的股票文件列表中获取股票编号列表，600006.ss
    def getSockNoList_File(self, sockListFile):
        titleNameList=['Code', 'Name', 'StartDate', 'amount0', 'amount1', 'postfix']
        
        dtype={'names': tuple(titleNameList), 'formats': (type(1), 'S10', 'S10', type(1.0), type(1.0), 'S10')}
        arr=np.loadtxt(sockListFile, dtype, comments='#', delimiter=',', converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)
        #由于设置了dtype，所以得出的是一个一维的数据，每一个元素都是原文件的一行的数据构成的元组
        data=[list(arr[rowNum]) for rowNum in range(arr.shape[0])]

        sub_sock_list=['0'*(6-len(str(item[0])))+str(item[0])+'.'+item[5] for item in data]
        
        return sub_sock_list
    
    def getSockNoList_Db(self, db_lock):
        res=self.DbEx_GetDataByTitle('summary_tb', ['itemName', 'itemValue'], db_lock, needSort=0)
        res_itemName=[item[0] for item in res]
        res_itemValue=[item[1] for item in res]
        res=res_itemValue[res_itemName.index('sockList')]
        return res.split(',')
    
    #------------------------------------------------------------------#
            
    def getExistDate_bySock(self, sockNo, db_lock):
        #获取每个股票的已经进入数据库的交易日列表
        bData=self.DbEx_GetDataByTitle(sockNo, ['Date', 'AdjClose'], db_lock)
        if bData.ndim==2:
            return bData[:,0].tolist()
        else:
            return []
        
    def getRowBy_BySockDate(self, sockNo, Titles, Date, db_lock):
        bData=self.DbEx_GetRowByTitle_ByDate(sockNo, Titles, Date, db_lock, outDataFormat=np.float32)
        return bData.tolist()
    
    def getTradeDateList_Network(self, locks, importDb=False):
        existDate=self.getTradeDateList_Db(locks['db_access'])
        
        if existDate:
            if int(max(existDate))==int(self.getTodayAsNum()):
                return existDate
        
        #获取所有的交易日日期
        sockNo='000001.ss'#上证综指代码

        #下载上证综指的全部交易记录
        res=self.downloadSockBaseData(sockNo, [])
        assert res
        
        csvFile=os.path.join(global_value.csvDir, sockNo+'.csv')
        
        #取出交易日信息
        Date=[]
        with open(csvFile, 'r') as pf:
            lines=pf.readlines()
            lines.pop(0)
            Date=[self.datestr2num(item.split(',')[0]) for item in lines]
        
        #排序以后返回
        Date = sorted(Date, reverse=True)
        
        if importDb:
            self.DbEx_UpdateItem('summary_tb', ['itemName', 'itemValue'], ['tradeDates', ','.join([str(int(date)) for date in Date])], locks['db_access'])
        
        return Date
    
    def getTradeDateList_Db(self, db_lock):
        res=self.DbEx_GetDataByTitle('summary_tb', ['itemName', 'itemValue'], db_lock, needSort=0)
        res_itemName=[item[0] for item in res]
        res_itemValue=[item[1] for item in res]
        res=res_itemValue[res_itemName.index('tradeDates')]
        res=res.split(',')
        try:
            res.remove('')
        except:
            pass
        res=[int(item) for item in res]
        return res

    #------------------------------------------------------------------#
    
    #下载某个股票的某些天的基本交易数据，用于数据初始化
    #如果DateList是一个空列表，就下载所有的交易数据
    #结果保存到csv文件中
    def downloadSockBaseData(self, sockNo, DateList):
        #构造URL并访问
        urlbase=global_value.yahoo_sock_url
    
        if not os.path.exists(global_value.csvDir):
            os.makedirs(global_value.csvDir)
    
        try:
            if DateList:
                s_y, s_m, s_d = self.num2ymd(min(DateList))
                e_y, e_m, e_d = self.num2ymd(max(DateList))
                conditionString='a=%d&b=%d&c=%d&d=%d&e=%d&f=%d' % (s_m-1, s_d, s_y, e_m-1, e_d, e_y)
                conditionString+='&'
            else:
                conditionString=''
                
            PROXY_ENABLE=global_value.config_proxy_en
            proxies={}
            if PROXY_ENABLE:
                a=global_value.config_proxy_user
                b=global_value.config_proxy_password
                if global_value.config_proxy_ip_http:
                    proxies['http']="http://"+a+":"+b+"@"+global_value.config_proxy_ip_http
                if global_value.config_proxy_ip_https:
                    proxies['https']="http://"+a+":"+b+"@"+global_value.config_proxy_ip_https                
                    
            r = requests.get(urlbase+conditionString+'s='+sockNo.replace('_', '.'), proxies=proxies)
                    
            with open(os.path.join(global_value.csvDir, sockNo+'.csv'), "wb") as pf:
                pf.write(r.content)
            #print 'download OK:'+sockNo
            return True
        except Exception as err:
            print (err)
            #print traceback.format_exc()
            #print 'download Fail:'+sockNo
            return False
        
    def readDataFromCSV(self, sockNo):
        csvFile=os.path.join(global_value.csvDir, sockNo+'.csv')
        
        titleNameList=global_value.column_Type_BaseData_Title
        
        if not os.path.isfile(csvFile):
            return titleNameList,[]      
        
        #read data from csv file
        dtype={'names': tuple(titleNameList), 'formats': (type(1), type(1.0), type(1.0), type(1.0), type(1.0), type(1.0), type(1.0))}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")        
            arr=np.loadtxt(csvFile, dtype, comments='#', delimiter=',', converters={0: self.datestr2num}, skiprows=1, usecols=None, unpack=False, ndmin=0)
        if not arr.shape:
            data=[list(arr.tolist())]
        else:
            data=[list(arr[rowNum]) for rowNum in range(arr.shape[0])]
            
        #剔除停盘日的数据
        data=[item for item in data if item[titleNameList.index('Volume')]>0.0]
        
        return titleNameList,data


    def getSockExtendData(self, allData):
        allColName=global_value.column_Type_BaseData_Title+global_value.column_Type_ExtendData_Title
        
        #涨幅和涨幅率
        #涨跌幅度，D日的涨幅是下一个交易日的收盘价减去D日的收盘价，不要弄错
        l=allData[:-1,allColName.index('AdjClose')]-allData[1:,allColName.index('AdjClose')]
        l=np.append(np.array([0]), l)
        lr=l/allData[:,allColName.index('AdjClose')]
        #insert data
        allData.T[allColName.index('PriceIncrease')]=l
        allData.T[allColName.index('PriceIncreaseRate')]=lr        
        
        
        #计算3,5,10,12,20,30,26,60日均值和均值率（比价格）
        allData=self.dataproc_getPriceMean(allColName, allData)
    
        #---------------------------------------------------------------------------------------
        #DIFF=EMA(12)-EMA(26):两条指数平滑移动平均线的差
        allData=self.dataproc_getDIFF_12_26(allColName, allData)
    
        #---------------------------------------------------------------------------------------
        #DEA:DIFF的M日的平均的指数平滑移动平均线,这里M=9
        allData=self.dataproc_getDEA(allColName, allData)
    
        #---------------------------------------------------------------------------------------
        #MACD:DIFF-DEA
        allData=self.dataproc_getMACD(allColName, allData)
    
        #---------------------------------------------------------------------------------------
        #KDJ，这里取n=9
        allData=self.dataproc_getKDJ(allColName, allData)
    
        #---------------------------------------------------------------------------------------
        #RSI相对强弱指标
        #RSI(N)计算方法：当天以前的N天的上涨日的涨幅和作为A，当天以前的N天的下跌日的跌幅幅和作为B（B取绝对值），RSI(N)=A/(A＋B)×100
        allData=self.dataproc_getRSI(allColName, allData)
    
        #---------------------------------------------------------------------------------------
        #BOLL布林线指标
        #中轨线=N日的移动平均线
        #上轨线=中轨线+两倍的标准差
        #下轨线=中轨线－两倍的标准差    
        #MA=N日内的收盘价之和÷N
        allData=self.dataproc_getBOLL(allColName, allData)
        
        #---------------------------------------------------------------------------------------
        #WR威廉指标--威廉超买超卖指数,主要用于分析市场短期买卖走势
        #计算方法：n日WMS=(Hn－Ct)/(Hn－Ln)×100,Ct为当天的收盘价；Hn和Ln是n日内（包括当天）出现的最高价和最低价
        allData=self.dataproc_getWR(allColName, allData)
        
        #---------------------------------------------------------------------------------------
        #DMI指标--DMI指标又叫动向指标或趋向指标
        #计算方法：
        allData=self.dataproc_getDMI(allColName, allData)        
    
        #---------------------------------------------------------------------------------------
        #主力买卖
        #EXPMA:指数平均数
        #大盘指数
        #横盘突破
        #振幅
        #委比
        

        
        return allData

    
    #每天的预测某个具体股票的函数
    def tipster_DailyProc(self, sockNo, locks):
        #利用已经得到的基本数据和扩展数据，进行kNN的系数权重训练
        
        
        #下面是要参考的信息项目        
        #ColTitleUsed=['Volume', 'mean_3_RatePrice', 'mean_5_RatePrice', 'mean_10_RatePrice', 'mean_20_RatePrice', 'mean_30_RatePrice', 'mean_60_RatePrice']
        #参考的数据，如Volume#3代表要参考Volume，而距离是使用3天曲线之间的距离。
        ColTitleUsed=['Volume#3', 'Volume#5', 'mean_3_RatePrice#5', 'mean_5_RatePrice#10', 'mean_10_RatePrice#20', 'mean_20_RatePrice#3', 'mean_30_RatePrice#3', 'mean_30_RatePrice#10']
        
        #从系数数据库中找出ColTitleUsed各个项对应的推荐系数，如果推荐系数不存在，就设置为1.0-------weight_db
        weightCoefficient_dist=self.getRecommondWeight(sockNo, locks['db_access'])
        weightCoefficient=[weightCoefficient_dist[item] if item in weightCoefficient_dist.keys() else 1.0 for item in ColTitleUsed]

        res=self.weightTrainning(ColTitleUsed, weightCoefficient, sockNo, locks['db_access'])
        #res是结构为[预测准确率, [系数序列]]的结果，表示在使用该系数序列情况下，能达到最高的预测准确率，预测率为"预测准确率"
        
        if not res:
            return
        
        #将'系数序列'做字典化以后，刷新到系数数据库中
        self.setRecommondWeight(sockNo, dict(zip(ColTitleUsed,res[1])), locks['db_access'])
     
        #利用已经训练得到的系数，进行预测    
        forecast_res=self.forecast_LastDate(ColTitleUsed, res[1], sockNo, locks['db_access'])
        #print "accuracy rate:",res[0]
        #print "growth rate:",forecast_res
        
        #将'预测准确率'和预测得到的涨幅放入数据库
        lastDate=self.DbEx_GetLastDate(sockNo, locks['db_access'])
        self.DbEx_UpdateItem(sockNo, ['Date', 'Forecast_Increase', 'Forecast_Accuracy'], [lastDate, forecast_res, res[0]], locks['db_access'])
        
        
    def getRecommondWeight(self, sockNo, lock):
        #read last weight from db file
        colName_sockNo='sockNo'
        colName_weight='weight_array'        
        
        self.DbEx_Connect(lock)
        query=self.DB_Base_Create_SqlCmd_SELECT('weight_tb', colName_weight, colName_sockNo+'="##0"', sockNo)
        res=self.DB_Base_Query(query)
        self.DbEx_Close(lock)

        if not res:
            return dict()
        else:
            assert len(res)==1
            temp_dict=json.loads(base64.b64decode(res[0][0]))
            res_dict=dict()
            for key in temp_dict.keys():
                res_dict[str(key)]=temp_dict[key]
            
            return res_dict

        
    def setRecommondWeight(self, sockNo, dict_weight, lock):
        
        colName_sockNo='sockNo'
        colName_weight='weight_array'
        
        #将weight_json(字符串)存入文件
        self.DbEx_Connect(lock)
        query=self.DB_Base_Create_SqlCmd_SELECT('weight_tb', colName_weight, colName_sockNo+'="##0"', sockNo)
        res=self.DB_Base_Query(query)
        self.DbEx_Close(lock)
        if not res:
            #insert
            self.DbEx_InsertItem('weight_tb', [colName_sockNo, colName_weight], [sockNo, base64.b64encode(json.dumps(dict_weight))], lock)
        else:
            #update
            self.DbEx_UpdateItem('weight_tb', [colName_sockNo, colName_weight], [sockNo, base64.b64encode(json.dumps(dict_weight))], lock)

        