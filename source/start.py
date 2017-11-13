# -*- coding: utf-8 -*-

#Python 2.7.x

#V0.01

import os
import re
import datetime
import time
import multiprocessing
import random
import numpy as np
import ConfigParser

import global_value
from CalcEngine_CPU import *
from version import Version
from InitSockData import InitSockData
from BaseFunc import BufPrint
from OtherTask import *
from Drawing import MyDrawing

#初始化全局变量
#这些全局变量统一被定义在一个模块中
#
def initGlobalValue():
    cf = ConfigParser.ConfigParser()
    cf.read("base.conf")
    
    global_value.g_threadnum_download=int(cf.get('performance', "threadnum_download"))
    global_value.g_threadnum_dataproc=int(cf.get('performance', "threadnum_dataproc"))
    global_value.g_threadnum_predict=int(cf.get('performance', "threadnum_predict"))
    
    global_value.g_opencl_accelerate=int(cf.get('hardware_accelerate', "opencl_accelerate"))
    
    #数据来源的url
    global_value.yahoo_sock_url=cf.get('base', "yahoo_sock_url").strip('"')+'?'
    
    #下载的数据的各个列的title
    global_value.downloadItemTitle=cf.get('base', "downloadItemTitle").strip('"').split(',')
    #日志
    global_value.loger=int(cf.get('base', "loger"))
    if global_value.loger:
        global_value.loger = LogRecorder()
    #csvDir
    global_value.csvDir=cf.get('base', "csvDir").strip('"')
    
    
    global_value.config_proxy_en=True if cf.get('network', "proxy_en")=='on' else False
    global_value.config_proxy_ip_http=cf.get('network', "proxy_ip_http")
    global_value.config_proxy_ip_https=cf.get('network', "proxy_ip_https")
    global_value.config_proxy_user=cf.get('network', "proxy_user")
    global_value.config_proxy_password=cf.get('network', "proxy_password")
    
    
    #扩展数据的
    #均值序列长度
    global_value.MEAN_LEN_LIST=cf.get('extend', "MEAN_LEN_LIST").strip('"').split(',')
    global_value.MEAN_LEN_LIST=[int(item) for item in global_value.MEAN_LEN_LIST]
    #需要做波动幅度的项目名称序列
    #global_value.FluctuateItem_LIST=cf.get('extend', "FluctuateItem_LIST").strip('"').split(',')
    #DEA_M
    global_value.DEA_M=int(cf.get('extend', "DEA_M"))
    #KDJ_N
    global_value.KDJ_N=int(cf.get('extend', "KDJ_N"))
    #RSI_DArray
    global_value.RSI_DArray=cf.get('extend', "RSI_DArray").strip('"').split(',')
    global_value.RSI_DArray=[int(item) for item in global_value.RSI_DArray]
    #BOLL_N
    global_value.BOLL_N=int(cf.get('extend', "BOLL_N"))
    #WR_DArray
    global_value.WR_DArray=cf.get('extend', "WR_DArray").strip('"').split(',')
    global_value.WR_DArray=[int(item) for item in global_value.WR_DArray]
    #DMI_DArray
    global_value.DMI_DArray=cf.get('extend', "DMI_DArray").strip('"').split(',')
    global_value.DMI_DArray=[int(item) for item in global_value.DMI_DArray]
    
    
    global_value.db_type=cf.get('db_info', "db_type").lower()
    
    if 'mysql'==global_value.db_type: #MySQL Database
        #数据库入口信息
        global_value.db_entry={}
        global_value.db_entry['server_ip']=cf.get('db_entry_mysql', "db_entry_server_ip")
        global_value.db_entry['port']=int(cf.get('db_entry_mysql', "db_entry_port"))
        global_value.db_entry['user']=cf.get('db_entry_mysql', "db_entry_user")
        global_value.db_entry['password']=cf.get('db_entry_mysql', "db_entry_password")
        global_value.db_entry['db_name']=cf.get('db_entry_mysql', "db_entry_db_name")
        
    elif 'sqlite'==global_value.db_type: #Sqlite Database
        global_value.db_entry={}
        global_value.db_entry['db_name']=cf.get('db_entry_sqlite', "db_entry_dbname")
        
        
    #每个sock会建立一个table,每个table的表项是一样的，如下
    global_value.column_Type_BaseData_Title=global_value.downloadItemTitle
    column_Type_BaseData_DataType=["int"]+["FLOAT"]*(len(global_value.column_Type_BaseData_Title)-1)
        
    global_value.column_Type_Forecast_Title=["Forecast_Increase", "Forecast_Accuracy"]
    column_Type_Forecast_DataType=["FLOAT"]*len(global_value.column_Type_Forecast_Title)
    
    
    global_value.column_Type_ExtendData_Title=["PriceIncrease", "PriceIncreaseRate"]
    for meanLen in global_value.MEAN_LEN_LIST:
        global_value.column_Type_ExtendData_Title+=['mean_'+str(meanLen), 'mean_'+str(meanLen)+'_RatePrice']
    #for item in global_value.FluctuateItem_LIST:
        #global_value.column_Type_ExtendData_Title+=["fluctuate_size0_"+item, "fluctuate_size0_"+item+"_Rate", "fluctuate_len0_"+item, "fluctuate_size1_"+item, "fluctuate_size1_"+item+"_Rate", "fluctuate_len1_"+item, "fluctuate_size2_"+item, "fluctuate_size2_"+item+"_Rate", "fluctuate_len2_"+item]
    global_value.column_Type_ExtendData_Title+=["DIFF_12_26", "DIFF_12_26_Rate"]
    global_value.column_Type_ExtendData_Title+=["DEA_"+str(global_value.DEA_M), "DEA_"+str(global_value.DEA_M)+"_Rate"]
    global_value.column_Type_ExtendData_Title+=["MACD", "MACD_Rate"]
    global_value.column_Type_ExtendData_Title+=["KDJ_K", "KDJ_D", "KDJ_J"]
    for item in global_value.RSI_DArray:
        global_value.column_Type_ExtendData_Title+=["RSI_"+str(item)]
    global_value.column_Type_ExtendData_Title+=["BOLL_MA_20", "BOLL_MA_20_Rate", "BOLL_UP_20", "BOLL_UP_20_Rate", "BOLL_DN_20", "BOLL_DN_20_Rate"]
    for item in global_value.WR_DArray:
        global_value.column_Type_ExtendData_Title+=["WR_"+str(item)]
    for item in global_value.DMI_DArray:
        global_value.column_Type_ExtendData_Title+=["DMI_PDI_"+str(item)]
        global_value.column_Type_ExtendData_Title+=["DMI_NDI_"+str(item)]
        global_value.column_Type_ExtendData_Title+=["DMI_ADX_"+str(item)]
        global_value.column_Type_ExtendData_Title+=["DMI_ADXR_"+str(item)]
        
    column_Type_ExtendData_DataType=["FLOAT"]*len(global_value.column_Type_ExtendData_Title)

    global_value.db_entry['table_construction']={'column': global_value.column_Type_BaseData_Title+global_value.column_Type_Forecast_Title+global_value.column_Type_ExtendData_Title, \
                                         'dataType':column_Type_BaseData_DataType+column_Type_Forecast_DataType+column_Type_ExtendData_DataType}
    global_value.db_entry['main_key']=global_value.column_Type_BaseData_Title[0]


if __name__ == '__main__':
    
    v=Version()
    print 'Tipster '+v.getVersionString()
    
    initGlobalValue()
    
    
    if global_value.g_opencl_accelerate:
        from CalcEngine_OpenCL import *
        
    
    print '---MODE 0:\tInit Database before any other process'
    print '---MODE 1:\tDownload ,proc and predict data of some socks (; to split)'
    print '---MODE 2:\tDownload ,proc and predict data of all socks'
    print '---MODE 8:\tShow info of one sock'
    print '---MODE 9:\tShow the claculate result'
    print '---MODE 10:\tTest System'
    

    mode=int(raw_input('run mode: '))
    
    
    global_value.g_curRunMode=mode
    
    if mode==10:#测试系统能力，主要是系统的GPU能力
        isThereGPU=False
        
        if global_value.g_opencl_accelerate:
            OC = OpenCL_Cap()
            if len(OC.handles):
                calcEngine = opencl_algorithms_engine(device_index=0, kernelFile='opencl/myKernels.cl')
                isThereGPU=True
                pass
            else:
                calcEngine = cpu_algorithms_engine()
                pass
        else:
            calcEngine = cpu_algorithms_engine()
            
        calcEngine.cl_algorithm_showClMemStruct()
        
        #检查GPU/OpenCL和CPU在算法实现上是否有差别。如果没有差别，平时可以调试CPU即可
        if isThereGPU:
            assert 0
            calcEngine_OpenCL = calcEngine
            calcEngine_CPU = cpu_algorithms_engine()
            
            dir_CE_OCL = dir(calcEngine_OpenCL)
            dir_CE_CPU = dir(calcEngine_CPU)
            
            dir_CE_OCL=[item for item in dir_CE_OCL if item.startswith('algorithm_')]
            dir_CE_CPU=[item for item in dir_CE_CPU if item.startswith('algorithm_')]
            
            #取不属于双方的algorithm_算法项
            diffAlgorithmSet=list(set(list(set(dir_CE_CPU).difference(set(dir_CE_OCL)))).union(set(list(set(dir_CE_OCL).difference(set(dir_CE_CPU))))))
            if len(diffAlgorithmSet):
                print 'diffAlgorithmSet:',diffAlgorithmSet
                print 'dir_CE_OCL:',dir_CE_OCL
                print 'dir_CE_CPU:',dir_CE_CPU
                assert 0
            
            #双方的算法集合一样。下面开始具体的测试CPU和OpenCL算法的一直性
            for item_algorithmFunc in dir_CE_OCL:
                if not hasattr(calcEngine_OpenCL, 'unitTest_'+item_algorithmFunc):
                    print 'Attr '+'unitTest_'+item_algorithmFunc+' is not exist'
                    continue
                ret = getattr(calcEngine_OpenCL, 'unitTest_'+item_algorithmFunc)(calcEngine_CPU, item_algorithmFunc)
                print item_algorithmFunc+' test: ', 'Pass' if ret else 'NO Pass'
            
            
        pass
    
    if mode==0:#初始化数据库，对全部sock增加相应的table
        init_handle=InitSockData()
        
        a=raw_input('####%%%%    Are you want to re-build the DataBase???(Type "yes" to continue)')
        if a.lower() != 'yes':
            exit()
        
        if 'mysql'==global_value.db_type:
            print 'use MySQL Database.'
        elif 'sqlite'==global_value.db_type:
            print 'use Sqlite Database.'
        else:
            assert 0
        
        resourceLocks={}
        resourceLocks['print']=multiprocessing.Lock()
        resourceLocks['db_access']=multiprocessing.Lock()          
        
        print 'drop database...'
        init_handle.__DbEx_DropDB__(resourceLocks['db_access'])
        
        print 'create database...'
        init_handle.__DbEx_CreateNewDB__(resourceLocks['db_access'])
        
        print 'get sock no list from CSV file...'
        sock_list=init_handle.getSockNoList_File(os.path.join('dir_baseData', 'sockList.csv'))
        print 'find '+str(len(sock_list))+' socks.'
        
        print 'create table for each sock...'
        init_handle.DbEx_Connect(resourceLocks['db_access'])
        for sockNo in sock_list:
            init_handle.__DbEx_CreateTable__(sockNo, global_value.db_entry['table_construction'])
        init_handle.DbEx_Commit()
        init_handle.DbEx_Close(resourceLocks['db_access'])
        print 'Create Tables OK'
        
        print 'create table for weight...'
        init_handle.DbEx_Connect(resourceLocks['db_access'])
        init_handle.__DbEx_CreateTable__('weight_tb', {'column': ['sockNo', 'weight_array'], 'dataType': ['CHAR(9)', 'TEXT']})
        init_handle.DbEx_Commit()
        init_handle.DbEx_Close(resourceLocks['db_access'])
        print 'create ok'
        
        print 'create table for Summary information, like sock list, tradeDate list...'
        init_handle.DbEx_Connect(resourceLocks['db_access'])
        init_handle.__DbEx_CreateTable__('summary_tb', {'column': ['itemName', 'itemValue'], 'dataType': ['CHAR(32)', 'LONGTEXT']})
        init_handle.DbEx_Commit()
        init_handle.DbEx_Close(resourceLocks['db_access'])
        
        print 'import sock list into summary_tb...'
        init_handle.DbEx_InsertItem('summary_tb', ['itemName', 'itemValue'], ['sockList', ','.join(sock_list)], resourceLocks['db_access'])
        init_handle.DbEx_InsertItem('summary_tb', ['itemName', 'itemValue'], ['tradeDates', ''], resourceLocks['db_access'])
        
        
        print 'Now the follow tables are create in DB:'
        init_handle.DbEx_Connect(resourceLocks['db_access'])
        db_tables = init_handle.__DbEx_GetTable__()
        init_handle.DbEx_Close(resourceLocks['db_access'])
        db_tables = [str(item[0]) for item in db_tables]
        db_tables_type_ss=[item for item in db_tables if re.match(r'''ss[0-9]{6}''', item)]
        db_tables_type_sz=[item for item in db_tables if re.match(r'''sz[0-9]{6}''', item)]
        db_tables_type_common=[item for item in db_tables if item not in db_tables_type_ss and item not in db_tables_type_sz]
        print '--------'
        print str(len(db_tables_type_ss))+' ss socks'
        print str(len(db_tables_type_sz))+' sz socks'
        print 'Other databse: '+','.join(db_tables_type_common)
        
        for table_name in db_tables_type_common:
            #显示数据库中的所有数据项名称
            names=init_handle.DbEx_GetColumns(table_name, resourceLocks['db_access'])
            print '--------'
            print 'Cols of '+table_name+' is:'
            print ','.join(names)
        if (len(db_tables_type_ss)+len(db_tables_type_sz)) >0:
            temp_list=db_tables_type_sz+db_tables_type_ss
            names=init_handle.DbEx_GetColumns(temp_list[0], resourceLocks['db_access'])
            print '--------'
            print 'Cols of common sock db is:'
            print ','.join(names)
        
        print '####&&&&to be finished####&&&&'
        
    elif mode==1:#下载，处理，和预测某个具体的sock
        sock=raw_input('input which sock to download and proc: ')
        init_handle=InitSockData()
        
        if 'mysql'==global_value.db_type:
            print 'use MySQL Database.'
        elif 'sqlite'==global_value.db_type:
            print 'use Sqlite Database.'
        else:
            assert 0
        
        resourceLocks={}
        resourceLocks['print']=multiprocessing.Lock()
        resourceLocks['db_access']=multiprocessing.Lock()
        
        if global_value.g_opencl_accelerate:
            OC = OpenCL_Cap()
            if len(OC.handles):
                calcEngine = opencl_algorithms_engine(device_index=0, kernelFile='opencl/myKernels.cl')
                pass
            else:
                calcEngine = cpu_algorithms_engine()
                pass
        else:
            calcEngine = cpu_algorithms_engine()
        
        tradeDateList=init_handle.getTradeDateList_Network(resourceLocks, importDb=True)
        print 'update trade date list ok'        
        
        start1=start2=start3=end=0
        res0=res1=res2=False
        start1=datetime.datetime.now()
        print 'Download Data'
        res0=InitDownload_Task_Core(0, calcEngine, sock, tradeDateList, resourceLocks)
        start2=datetime.datetime.now()
        print 'Import Data into DB'
        res1=InitDataProc_Task_Core(0, calcEngine, sock, resourceLocks)
        start3=datetime.datetime.now()
        print 'do forecast'
        res2=InitTipsterProc_Task_Core(0, calcEngine, sock, resourceLocks)
        end=datetime.datetime.now()
        print sock+' :'+('True' if res0 else 'False')+' '+('True' if res1 else 'False')+' '+('True' if res2 else 'False')
        print 'Time:'
        print start1
        print start2
        print start3
        print end        
        
    elif mode==2:
        
        init_handle=InitSockData()
        
        if 'mysql'==global_value.db_type:
            print 'use MySQL Database.'
        elif 'sqlite'==global_value.db_type:
            print 'use Sqlite Database.'
        else:
            assert 0
            
        resourceLocks={}
        resourceLocks['print']=multiprocessing.Lock()
        resourceLocks['db_access']=multiprocessing.Lock()            
            
        init_handle.getTradeDateList_Network(resourceLocks, importDb=True)
        print 'update trade date list ok'
        
        start1=start2=start3=end=0
        raw_input('Notice: "multiprocessing" is not run very well on Wing IDE(Press any key to continue):')
        multiprocessing.freeze_support()
        start1=datetime.datetime.now()
        multiprocessing.freeze_support()
        Init_Download(global_value.g_threadnum_download, resourceLocks)
        start2=datetime.datetime.now()
        Init_DataProc(global_value.g_threadnum_dataproc, resourceLocks)
        start3=datetime.datetime.now()
        Init_TipsterProc(global_value.g_threadnum_predict, resourceLocks)
        end=datetime.datetime.now()
        print 'Time:'
        print start1
        print start2
        print start3
        print end
        
        
    elif mode==7:
        md = MyDrawing()
        md.example()
        print 'Proc Over'
        
    elif mode==8:
        #显示某个股票的各种信息
        sock=raw_input('input which sock to download and proc: ')
        init_handle=InitSockData()
        
        resourceLocks={}
        resourceLocks['print']=multiprocessing.Lock()
        resourceLocks['db_access']=multiprocessing.Lock()
        
        #显示数据库中的所有数据项名称
        names=init_handle.DbEx_GetColumns(sock, resourceLocks['db_access'])
        print 'All column names are: ',','.join(names)
        
        #打印这个股票的预测涨幅和预测准确率
        bData=init_handle.DbEx_GetDataByTitle(sock, ['Date', 'Forecast_Increase', 'Forecast_Accuracy'], resourceLocks['db_access'], outDataFormat=np.float32)
        
        tf=lambda x:init_handle.num2datestr(int(bData[x,0]))+'<>'+str(round(bData[x,1], 4)*100)+'%'+'<>'+str(bData[x,2])
        print tf(0)
        print tf(1)
        print tf(2)
        
        #计算这个股票历史的股票增长概率，即过去增长的天数占比，增长天数的统计又区分是否将0增长记为增长还是不增长
        bData=init_handle.DbEx_GetDataByTitle(sock, ['Date', 'PriceIncreaseRate'], resourceLocks['db_access'], outDataFormat=np.float32)
        t1=np.array([1 if bData[i,1]>0.0 else 0 for i in range(bData.shape[0])]).sum()
        t2=np.array([1 if bData[i,1]==0.0 else 0 for i in range(bData.shape[0])]).sum()
        print 'history increase date percent:',str(round((float(t1)/bData.shape[0])*100, 1)),'%-',str(round((float(t1+t2)/bData.shape[0])*100, 1)),'%'

        pass
        
    
    elif mode==9:
        N_days=20
        init_handle=InitSockData()
        
        resourceLocks={}
        resourceLocks['print']=multiprocessing.Lock()
        resourceLocks['db_access']=multiprocessing.Lock()        
            
        date_show = raw_input('Input the date to show(YYYYMMDD):')
        if not re.match(r'''[0-9]{8}''', date_show) or len(date_show)!=len('YYYYMMDD'):
            assert 0
        
        #从数据库中导出所有的结果，以特定的策略显示，比如按预测准确率最高的
        bData=[]
        for sockNo in sockList:
            bData_temp=init_handle.getRowBy_BySockDate(sockNo, ['Forecast_Increase', 'Forecast_Accuracy'], date_show[0:4]+'-'+date_show[4:6]+'-'+date_show[6:], db_lock, outDataFormat=np.float32)
            if bData_temp:
                bData.append([sockNo]+bData_temp+[bData_temp[0]*bData_temp[1]])
                pass
            pass
        
        if len(bData)<2*N_days:
            assert 0
        
        bp=BufPrint()
        #1.以预测准确率排序，导出最高的N个股票
        bp.BufPrint ('sort by Forecast_Accuracy')
        bp.BufPrint ('sockNo',':\t','ForeInc',',\t','ForeAcc',',\t','mean')
        bData.sort(lambda x,y: cmp(x[2],y[2]), reverse=True)    #降序排列
        for i in range(N_days):
            bp.BufPrint (bData[0][0],':\t',bData[0][1],',\t',bData[0][2],',\t',bData[0][3])
            bp.BufPrint ('...')
            pass
        bData.sort(lambda x,y: cmp(x[2],y[2]), reverse=False)    #升序排列
        for i in range(N_days):
            bp.BufPrint (bData[0][0],':\t',bData[0][1],',\t',bData[0][2],',\t',bData[0][3])
            pass
        
        #2.以预测涨幅排序，导出最高的N个股票
        bp.BufPrint ('sort by Forecast_Increase')
        bp.BufPrint ('sockNo',':\t','ForeInc',',\t','ForeAcc',',\t','mean')
        bData.sort(lambda x,y: cmp(x[1],y[1]), reverse=True)    #降序排列
        for i in range(N_days):
            bp.BufPrint (bData[0][0],':\t',bData[0][1],',\t',bData[0][2],',\t',bData[0][3])
            bp.BufPrint ('...')
            pass
        bData.sort(lambda x,y: cmp(x[1],y[1]), reverse=False)    #升序排列
        for i in range(N_days):
            bp.BufPrint (bData[0][0],':\t',bData[0][1],',\t',bData[0][2],',\t',bData[0][3])
            pass
        
        #3.以涨幅数学期望（预测涨幅和预测准确率乘积）排序，导出最高的N个股票
        bp.BufPrint ('sort by Forecast_Accuracy')
        bp.BufPrint ('sockNo',':\t','ForeInc',',\t','ForeAcc',',\t','mean')        
        bData.sort(lambda x,y: cmp(x[3],y[3]), reverse=True)    #降序排列
        for i in range(N_days):
            bp.BufPrint (bData[0][0],':\t',bData[0][1],',\t',bData[0][2],',\t',bData[0][3])
            bp.BufPrint ('...')
            pass
        bData.sort(lambda x,y: cmp(x[3],y[3]), reverse=False)    #升序排列
        for i in range(N_days):
            bp.BufPrint (bData[0][0],':\t',bData[0][1],',\t',bData[0][2],',\t',bData[0][3])
            pass
        
        
        #bp.getBuffer()
    
        pass
    
