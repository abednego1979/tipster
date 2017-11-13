# -*- coding: utf-8 -*-

#Python 2.7.x

#V0.01

import os
import datetime
import time
import multiprocessing
import random
import numpy as np

import global_value
from BaseVariable import BaseVariable
from BaseFunc import BaseFunc
from BaseFunc import LogRecorder
from InitSockData import InitSockData
from version import Version


#暂时用来做测试代码

__metaclass__ = type

#通用多任务处理模型1
#创建N和任务，N+1个队列
#主任务通过N个队列向N个任务发送具体的一个个任务，多个任务通过1个队列向主任务返回处理结果
class commonMultiTask_Model01():
    maxTaskNum=1
    que_main=None
    que_sub=None
    hook_MainTaskSndQue_Func=None
    hook_MainTaskRcvQue_Func=None
    hook_MainTaskSndQue_Args=None
    hook_MainTaskRcvQue_Args=None
    
    def __init__(self, maxTaskNum=1):
        self.maxTaskNum=maxTaskNum
        self.que_main=multiprocessing.Queue()
        self.que_sub=multiprocessing.Queue()
            
    def __del__(self):
        self.que_main.close()
        self.que_sub.close()
            
    def setHook_MainTaskSndQue(self, func, argDict):
        self.hook_MainTaskSndQue_Func=func
        self.hook_MainTaskSndQue_Args=argDict
        return
    
    def setHook_MainTaskRcvQue(self, func, argDict):
        self.hook_MainTaskRcvQue_Func=func
        self.hook_MainTaskRcvQue_Args=argDict
        return
    

    def MultiTaskStart(self, target, args, sub_paralist):
        
        #创建打印锁
        resourceLocks={}
        resourceLocks['print']=multiprocessing.Lock()
        resourceLocks['db_access']=multiprocessing.Lock()
        
        #创建多个任务
        task_array=[]
        for i in range(self.maxTaskNum):
            multiprocessing.freeze_support()
            #创建计算引擎（类示例）calcEngine
            OC = OpenCL_Cap()
            if len(OC.handles):
                calcEngine = opencl_algorithms_engine(device_index=0, kernelFile='')
                pass
            else:
                calcEngine = cpu_algorithms_engine()
                pass
            
            para=[i, self.que_sub, self.que_main, calcEngine, resourceLocks]+list(args)
            p=multiprocessing.Process(target=target, args=tuple(para))
            task_array.append(p)
            p.start()
        #将任务分配给各个子任务，然后监听子任务发送回来的数据并记录。同时读取各个子任务的待完成下载任务数量，进行任务间平衡
        #最后当下载任务结束后，向子任务发送停止运行的指令，并等待子任务结束
        sock_index=-1
        while sub_paralist:
            sub_item = sub_paralist.pop(0)
            sock_index+=1
            m=['proc', sub_item]
            try:
                if self.hook_MainTaskSndQue_Func:
                    self.hook_MainTaskSndQue_Func(sock_index%self.maxTaskNum, m, self.hook_MainTaskSndQue_Args, resourceLocks['print'])
            except:
                pass
            self.que_sub.put(m)
        #在队列中加入足够数量的毒药药丸，用于子任务在处理结束以后自杀
        for i in range(self.maxTaskNum):
            self.que_sub.put(['stop', ''])
        
        while True:
            if self.que_sub.qsize():
                #如果子任务们还有数据没处理，说明子任务还在运行
                if self.que_main.qsize():
                    message=self.que_main.get()
                    if message[0]=='proc_ok':
                        try:
                            if self.hook_MainTaskRcvQue_Func:
                                self.hook_MainTaskRcvQue_Func(None, message, self.hook_MainTaskRcvQue_Args, resourceLocks['print'])
                        except:
                            pass
                    else:
                        try:
                            if self.hook_MainTaskRcvQue_Func:
                                self.hook_MainTaskRcvQue_Func(None, message, self.hook_MainTaskRcvQue_Args, resourceLocks['print'])
                        except:
                            pass
            else:#如果子任务要处理的数据队列已经空了，说明子任务已经自杀，或者至少已经读取了毒丸，所以主任务延时若干秒以后结束
                time.sleep(5)
                break
            
        for i in range(self.maxTaskNum):
            #回收计算引擎
            calcEngine.delete_engine()
            pass
                
        return

    
def lockPrint(Tid, lock, outString):
    lock.acquire()
    print 'T'+str(Tid)+'\t: '+outString
    lock.release()

######下载所有的历史数据,并导入数据库##############################################################################
#下载所有的股票信息
#这个函数只下载数据到csv文件，并不处理和向数据库导入数据
def InitDownload_Task_Core(Tid, calcEngine, sockNo, tradeDateList, locks):
    DownloadFailNum=0
    proc_ok_flag=False
    
    initObj=InitSockData(calcEngine=calcEngine)
    csvFile=os.path.join(global_value.csvDir, sockNo+'.csv')
    try:
        os.remove(csvFile)#删除以前下载的csv文件
    except:
        pass
    
    while True:
        #得到数据库中已经存在的数据的日期
        existDateList=initObj.getExistDate_bySock(sockNo, locks['db_access'])
        existDateList.append(0) #为了方便下一步计算
        #数据库中还缺少的的数据的日期
        lackDateList=[item for item in tradeDateList if item > max(existDateList)]
        existDateList.remove(0)
        
        if not lackDateList:
            #如果数据库中的数据已经是最新的
            proc_ok_flag=True
            break
        else:
            #下载缺失的数据
            if len(existDateList):
                #数据库中已经有这个sock的数据，现在需要补充下载
                res=initObj.downloadSockBaseData(sockNo, lackDateList)
            else:
                #全新下载
                res=initObj.downloadSockBaseData(sockNo, [])
                
            if res:#downoad ok
                lockPrint(Tid, locks['print'], sockNo+' download OK')
                proc_ok_flag=True
                break
            else:
                lockPrint(Tid, locks['print'], sockNo+' download data fail')
                DownloadFailNum+=1
                if DownloadFailNum>=3:
                    break
                else:
                    continue
    #返回
    return proc_ok_flag
    

def InitDownload_Task(taskID, inQue, outQue, calcEngine, locks, obj, tradeDateList):    
    lockPrint(taskID, locks['print'], 'Task:'+str(taskID)+' Start')
    while True:
        m=inQue.get()
        
        if m[0]=='proc':
            lockPrint(taskID, locks['print'], 'Task:'+str(taskID)+' start to download '+m[1])
            
            if False:
                time.sleep(random.uniform(2, 12))
            else:
                time.sleep(random.uniform(1, 3))
                res=InitDownload_Task_Core(taskID, calcEngine, m[1], tradeDateList, locks)                   
            
            lockPrint(taskID, locks['print'], 'Task:'+str(taskID)+' end to download '+m[1])
            
            if res:
                outQue.put(['proc_ok', m[1]])
            else:
                outQue.put(['proc_fail', m[1]])
                
        elif m[0]=='stop':
            lockPrint(taskID, locks['print'], 'Task:'+str(taskID)+' goto stop')
            break
        else:
            assert 0,m
            return
    return

def InitDownload_Task_hookMainTask_procQueSnd(TaskId, message, argsDict, lock):
    return

def InitDownload_Task_hookMainTask_procQueRcv(TaskId, message, argsDict, lock):
    #记录哪些下载任务下载失败
    #
    if message[0]=='proc_ok':
        #将message[1]写入文件保存
        try:
            with open(argsDict['printFileName'], 'a') as pf:
                pf.write(message[1]+' download ok '+ global_value.sysEnterChar)
        except:
            pass        
    elif message[0]=='proc_fail':
        #将message[1]写入文件保存
        try:
            with open(argsDict['printFileName'], 'a') as pf:
                pf.write(message[1]+' download fail '+ global_value.sysEnterChar)
        except:
            pass
    return

def Init_Download(maxTaskNum, locks):
    init_handle=InitSockData()
    
    #获取要处理的股票编码列表
    sub_sock_list=init_handle.getSockNoList_Db(locks['db_access'])
    
    #获取当前最新的交易日信息
    tradeDateList=init_handle.getTradeDateList_Db(locks['db_access'])
    
    mt=commonMultiTask_Model01(maxTaskNum)
    mt.setHook_MainTaskSndQue(InitDownload_Task_hookMainTask_procQueSnd, {})
    mt.setHook_MainTaskRcvQue(InitDownload_Task_hookMainTask_procQueRcv, {'printFileName':'proc_over.dat'})
    try:
        with open('proc_over.dat', 'a') as pf:
            pf.write(global_value.sysEnterChar)
            pf.write(str(datetime.datetime.now()))
            pf.write(global_value.sysEnterChar)
    except:
        pass
    mt.MultiTaskStart(InitDownload_Task, (init_handle, tradeDateList), sub_sock_list)

    print 'download over'
####################################################################################


####对数据库中的数据计算得到扩展数据###########################################
#基本处理方式:本函数每次取一次历史数据，然后计算一天的数据，然后插入数据库。每一天的数据需要耗时1-2秒，对于数据库的初时计算非常不利 
#改进方式1:除了在开始从数据库读取数据，和最后将数据写入数据库，中间的数据处理全在内存中进行
#改进方式2:在改进方式1的基础上，如果要处理的数据过多，比如超过30条，就以全重新生成的方式计算（由于全重新生成可以使用numpy计算，速度比较快），缺点是数据处理要分成逐天处理和全部重计算两个函数
#改进方式3:在改进方式1的基础上，以全重新生成的方式计算，缺点是每天例行计算时都需要全部计算，时间比不改进的方案的每条计算量大
#改进方式4:在改进方式1的基础上，用要处理数据，拼接上过去N天的数据，进行重新计算。这样做是考虑到股票数据数据都是基于过去N天以内的数据进行计算的，更早的数据实际上对现在的数据没有影响

def InitDataProc_Task_Core(Tid, calcEngine, sockNo, locks):
    return InitDataProc_Task_Core_Mode03(Tid, calcEngine, sockNo, locks)


#处理方式3:一开始重数据库中中读取全部数据，用待新增计算的前N天的数据配合完成对新增数据的计算，最后将新增数据插入到数据库
#N暂定为80
def InitDataProc_Task_Core_Mode03(Tid, calcEngine, sockNo, locks):
    preDays_N=80
    
    
    initObj=InitSockData(calcEngine=calcEngine)
    allColName=global_value.column_Type_BaseData_Title+global_value.column_Type_ExtendData_Title
    
    #打开文件，读取所有的数据
    fTitle, fData=initObj.readDataFromCSV(sockNo)#fData是列表形式
    if len(fData) == 0:
        return True
    #得到所有要更新的日期信息，以后备用
    insertItemNum=len(fData)
    
    #fData转换为array，并转换为2维数据
    fData=np.array(fData)
    if fData.ndim==1:
        fData.shape=(1, fData.shape[0])
    
    #对文件中读取的数据，依据日期进行降序排序
    x=fData.T.argsort()
    fData=np.array([fData[x[0].tolist()[::-1][index],:].tolist() for index in range(x.shape[1])])
    
    #将fData补齐到与数据库的列数一样
    fData=initObj.npColMerge((fData, np.zeros((fData.shape[0],len(allColName)-fData.shape[1]))))
    
    #从数据库中读取所有的信息，并将新数据和数据库的数据结合
    fData=initObj.npRowMerge((fData, initObj.DbEx_GetDataByTitle(sockNo, allColName, locks['db_access'])))
    
    #取pData的最新的fData.shape[0]+preDays_N条记录
    fData=fData[:(insertItemNum+preDays_N),:]
    if fData.shape[0]<(insertItemNum+preDays_N):
        #如果数据库已经存在的数据不够preDays_N条，则用最老的一条数据进行补齐***************************************
        more_item=np.array([fData[-1,:].tolist()]*(insertItemNum+preDays_N-fData.shape[0]))
        fData=initObj.npRowMerge((fData, more_item))
    
    #用fData进行批量的扩展数据计算
    fData=initObj.getSockExtendData(fData)
    
    
    lastDataItem=fData[insertItemNum, :]    #本次刷新前的最后一条记录
    fData=fData[:insertItemNum, :]          #本次增加的记录
    
    #由于计算的结果可能出现nan值，这种值后继是不能插入数据库的，所以要对是否存在nan值做检查
    assert not np.isnan(fData).any()
    assert not np.isnan(lastDataItem).any()
    
    
    if fData[-1,allColName.index('Date')]==lastDataItem[allColName.index('Date')]:
        #如果本次要刷新的记录的最后一条信息的日期与刷新前最后一条记录的日期相同，说明本次刷新前数据库中就没有数据，
        #所谓的刷新前最后一条记录是前面程序扩展的,见前面注释中*号较多的代码
        pass
    else:
        #上一次计算的数据库的最后一条记录的涨幅和涨幅率需要更新
        lastDate=lastDataItem.tolist()[allColName.index('Date')]
        lastPriceRaise=lastDataItem.tolist()[allColName.index('PriceIncrease')]
        lastPriceRaiseRate=lastDataItem.tolist()[allColName.index('PriceIncreaseRate')]
        
        initObj.DbEx_UpdateItem(sockNo, ['Date', 'PriceIncrease', 'PriceIncrease'], [lastDate, lastPriceRaise, lastPriceRaiseRate], locks['db_access'], needConnect=True, needCommit=True)
    
    initObj.DbEx_Connect(locks['db_access'])
    for i in range(fData.shape[0]):
        initObj.DbEx_InsertItem(sockNo, allColName, fData[i].tolist(), locks['db_access'], needConnect=False, needCommit=False)
    initObj.DbEx_Commit()
    initObj.DbEx_Close(locks['db_access'])
    
    return True
        
    

    
def Init_DataProc_Task(taskID, inQue, outQue, calcEngine, locks, obj):
    lockPrint(taskID, locks['print'], 'Task:'+str(taskID)+' Start')
    
    while True:
        m=inQue.get()
        
        if m[0]=='proc':
            lockPrint(taskID, locks['print'], 'Task:'+str(taskID)+' start to proc '+m[1])
            
            if False:
                time.sleep(random.uniform(2, 12))
            else:
                time.sleep(random.uniform(1, 3))
                res=InitDataProc_Task_Core(taskID, calcEngine, m[1], locks)
                if not res:
                    try:
                        os.remove(os.path.join(obj.dbDir, sockNo+'.sqlite'))
                    except:
                        pass
            
            lockPrint(taskID, locks['print'], 'Task:'+str(taskID)+' end to proc '+m[1])
            
            if res:
                outQue.put(['proc_ok', m[1]])
            else:
                outQue.put(['proc_fail', m[1]])
        elif m[0]=='stop':
            lockPrint(taskID, locks['print'], 'Task:'+str(taskID)+' goto stop')
            break
        else:
            assert 0,m
            return
    return

def InitDataProc_Task_hookMainTask_procQueSnd(TaskId, message, argsDict, lock):
    return

def InitDataProc_Task_hookMainTask_procQueRcv(TaskId, message, argsDict, lock):
    #记录哪些处理任务失败
    #
    if message[0]=='proc_ok':
        #将message[1]写入文件保存
        try:
            with open(argsDict['printFileName'], 'a') as pf:
                pf.write(message[1]+' proc ok '+ global_value.sysEnterChar)
        except:
            pass        
    elif message[0]=='proc_fail':
        #将message[1]写入文件保存
        try:
            with open(argsDict['printFileName'], 'a') as pf:
                pf.write(message[1]+' proc fail '+ global_value.sysEnterChar)
        except:
            pass
    return
    
def Init_DataProc(maxTaskNum, locks):
    init_handle=InitSockData()
    
    sub_sock_list=init_handle.getSockNoList_Db(locks['db_access'])
    #sub_sock_list=sub_sock_list[:100]
    
    #下面用多任务方式处理
    mt=commonMultiTask_Model01(maxTaskNum)
    mt.setHook_MainTaskSndQue(InitDataProc_Task_hookMainTask_procQueSnd, {})
    mt.setHook_MainTaskRcvQue(InitDataProc_Task_hookMainTask_procQueRcv, {'printFileName':'proc_over.dat'})
    try:
        with open('proc_over.dat', 'a') as pf:
            pf.write( global_value.sysEnterChar ) 
            pf.write(str(datetime.datetime.now()))
            pf.write( global_value.sysEnterChar )
    except:
        pass
    mt.MultiTaskStart(Init_DataProc_Task, [init_handle], sub_sock_list)
    
    print 'Proc Over'
####################################################################################
    

####对数据库中的数据计算预测###########################################
def InitTipsterProc_Task_Core(Tid, calcEngine, sockNo, locks):
    proc_ok_flag=False
    
    initObj=InitSockData(calcEngine=calcEngine)

    #进行预测
    initObj.tipster_DailyProc(sockNo, locks)
    
            
    proc_ok_flag=True
    
    return proc_ok_flag


def Init_TipsterProc_Task(taskID, inQue, outQue, calcEngine, locks, obj):
    lockPrint(taskID, locks['print'], 'Task:'+str(taskID)+' Start')
    
    while True:
        m=inQue.get()
        
        if m[0]=='proc':
            lockPrint(taskID, locks['print'], 'Task:'+str(taskID)+' start to predict '+m[1])
            
            if False:
                time.sleep(random.uniform(2, 12))
            else:
                time.sleep(random.uniform(1, 3))
                res=InitTipsterProc_Task_Core(taskID, calcEngine, m[1], locks)
            
            lockPrint(taskID, locks['print'], 'Task:'+str(taskID)+' end to predict '+m[1])
            
            if res:
                outQue.put(['proc_ok', m[1]])
            else:
                outQue.put(['proc_fail', m[1]])
        elif m[0]=='stop':
            lockPrint(taskID, locks['print'], 'Task:'+str(taskID)+' goto stop')
            break
        else:
            assert 0,m
            return
    return


def InitTipsterProc_Task_hookMainTask_procQueSnd(TaskId, message, argsDict, lock):
    return

def InitTipsterProc_Task_hookMainTask_procQueRcv(TaskId, message, argsDict, lock):
    #记录哪些处理任务失败
    #
    if message[0]=='proc_ok':
        #将message[1]写入文件保存
        try:
            with open(argsDict['printFileName'], 'a') as pf:
                pf.write(message[1]+' predict ok '+'\n')
        except:
            pass        
    elif message[0]=='proc_fail':
        #将message[1]写入文件保存
        try:
            with open(argsDict['printFileName'], 'a') as pf:
                pf.write(message[1]+' predict fail '+'\n')
        except:
            pass
    return

def Init_TipsterProc(maxTaskNum, locks):
    init_handle=InitSockData()
    
    sub_sock_list=init_handle.getSockNoList_Db(locks['db_access'])
    #sub_sock_list=sub_sock_list[:100]
    
    #下面用多任务方式处理
    mt=commonMultiTask_Model01(maxTaskNum)
    mt.setHook_MainTaskSndQue(InitTipsterProc_Task_hookMainTask_procQueSnd, {})
    mt.setHook_MainTaskRcvQue(InitTipsterProc_Task_hookMainTask_procQueRcv, {'printFileName':'proc_over.dat'})
    try:
        with open('proc_over.dat', 'a') as pf:
            pf.write( global_value.sysEnterChar )
            pf.write(str(datetime.datetime.now()))
            pf.write( global_value.sysEnterChar )
    except:
        pass
    mt.MultiTaskStart(Init_TipsterProc_Task, [init_handle], sub_sock_list)
    
    print 'Proc Over'
####################################################################################

    

