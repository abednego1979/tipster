# -*- coding: utf-8 -*-

#Python 2.7.x

import platform


g_curRunMode=None

g_threadnum_download=1
g_threadnum_dataproc=1
g_threadnum_predict=1

g_opencl_accelerate=0


sysEnterChar=''
if platform.system() == "Windows":
    sysEnterChar='\r\n'
elif platform.system() == "Linux":
    sysEnterChar='\n'
else:#for mac os
    sysEnterChar='\r'
#数据来源的url
yahoo_sock_url=''

#下载的数据的各个列的title
downloadItemTitle=''
#日志
loger=''
#csvDir
csvDir=''

#proxy相关
config_proxy_en=None
config_proxy_ip_http=None
config_proxy_ip_https=None
config_proxy_user=None
config_proxy_password=None



#扩展数据的
#均值序列长度
MEAN_LEN_LIST=[]
#需要做波动幅度的项目名称序列
FluctuateItem_LIST=[]
#DEA_M
DEA_M=0
#KDJ_N
KDJ_N=0
#RSI_DArray
RSI_DArray=[]
#BOLL_N
BOLL_N=0
#WR_DArray
WR_DArray = []
#DMI_DArray
DMI_DArray = []


db_type=0   #0-MySql, 1-Sqlite

db_entry={}


column_Type_BaseData_Title=[]
column_Type_Forecast_Title=[]
column_Type_ExtendData_Title=[]
