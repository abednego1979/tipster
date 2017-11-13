# -*- coding: utf-8 -*-

#Python 2.7.x

#V0.01

import traceback
import datetime
import os
import copy
import re
import sqlite3

from DB_Base import MyDB_Base

__metaclass__ = type

#class database
class MyDB_Sqlite(MyDB_Base):
    #重载的函数
    
    def __init__(self):
        MyDB_Base.__init__()
        
    def DB_Sqlite_Connect(self, db_entry, db_lock):
        db_lock.acquire()
        self.db_conn=sqlite3.connect(db_entry['db_name']+'.sqlite')
        self.db_curs=self.db_conn.cursor()
        #指定数据库内部是utf-8编码的
        self.db_conn.text_factory=str
        return


    def DB_Sqlite_CreateDB(self, db_entry, db_lock):
        #创建一个新的数据库文件即可（通过sqlite3的connect方法）
        self.DB_Sqlite_Connect(db_entry, db_lock)
        self.DB_Base_Close(db_lock)
        return
        
    def DB_Sqlite_DropDB(self, db_entry):
        #删除相应的数据库文件即可
        try:
            os.remove(db_entry['db_name']+'.sqlite')
        except:
            pass            
        return
    
    def DB_Sqlite_Create_SqlCmd_INSERT(self, tb_name, titleNameList, data):
        insert='INSERT INTO '+self.__DbEx_TransSockNo2SockName__(tb_name)+' ('+','.join(titleNameList)+') VALUES ('
        for value in data:
            para1=value
            if type(para1)==str:
                para1='"'+para1+'"'
            insert+=str(para1)+','
        insert=insert.rstrip(',')
        insert+=')'

        return insert
        