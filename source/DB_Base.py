# -*- coding: utf-8 -*-

#Python 2.7.x

#V0.01

import traceback
import os
import copy
import re

import global_value

__metaclass__ = type

#class database
class MyDB_Base():
    db_conn=None
    db_curs=None
    #数据库操作系列函数
    
    def __init__(self):
        db_conn=None
        db_curs=None
        return
    
    def DB_Base_CreateDB(self, db_entry, db_lock):
        if 'mysql'==global_value.db_type:
            self.DB_MySql_CreateDB(global_value.db_entry)
        elif 'sqlite'==global_value.db_type:
            self.DB_Sqlite_CreateDB(global_value.db_entry, db_lock)
        else:
            assert 0
        return
        
    def DB_Base_DropDB(self, db_entry):
        if 'mysql'==global_value.db_type:
            self.DB_MySql_DropDB(global_value.db_entry)
        elif 'sqlite'==global_value.db_type:
            self.DB_Sqlite_DropDB(global_value.db_entry)
        else:
            assert 0
        return
    
    def DB_Base_Connect(self, db_name, dbLock):
        if 'mysql'==global_value.db_type:
            self.DB_MySql_Connect(global_value.db_entry)
        elif 'sqlite'==global_value.db_type:
            self.DB_Sqlite_Connect(global_value.db_entry, dbLock)
        else:
            assert 0
        return
        
    def DB_Base_Commit(self):
        self.db_conn.commit()
        
    def DB_Base_Close(self, dbLock):
        self.db_curs.close()
        self.db_conn.close()
        self.db_curs=None
        self.db_conn=None
        if 'sqlite'==global_value.db_type:
            dbLock.release()
        
        
    ####################################################    
    def DB_Base_CreateTable(self, tb_name, tableConstruct, auto_increment):
        # create tables
        execString="create table "+tb_name+" ("
        
        for index,title_type in enumerate(zip(tableConstruct['column'], tableConstruct['dataType'])):
            execString+=title_type[0]+' '+title_type[1]
            if index==0:
                if auto_increment:
                    execString+=' PRIMARY KEY NOT NULL AUTO_INCREMENT'
                else:
                    execString+=' PRIMARY KEY NOT NULL'
            execString+=','
        execString=execString.rstrip(',')
        execString+=')'
        try:
            self.db_curs.execute(execString)
        except Exception as err:
            print('The table '+tb_name+' exists!')
            print (err)
            print traceback.format_exc()
        return
    
    def DB_Base_GetTable(self):
        #get all tables name
        queryString = 'SHOW TABLES'
        try:
            self.db_curs.execute(queryString)
            l_temp=[]
            for row in self.db_curs.fetchall():
                l_temp.append(copy.deepcopy(row.values()))
            return l_temp            
        except Exception as err:
            print('SHOW TABLES Fail')
            print (err)
            print traceback.format_exc()       
            return []
    
    def DB_Base_DropTable(self, tb_name):
        # drop table
        execString='drop table '+tb_name
        try:
            self.db_curs.execute(execString)
        except Exception as err:
            print (err)
            print traceback.format_exc()
        return
        
    def DB_Base_Create_SqlCmd_SELECT(self, tb_name, sql_select, sql_where, *para):
        if len(sql_where):
            str_select='SELECT '+sql_select+' FROM '+tb_name+' WHERE '+sql_where
        else:
            str_select='SELECT '+sql_select+' FROM '+tb_name
            
        for i in range(len(para)):
            s=re.search(r"(?<=')"+'##'+str(i)+r"(?=')", str_select)
            str_select=str_select.replace('##'+str(i), str(para[i]))
        return str_select
    
    def DB_Base_Create_SqlCmd_UPDATE(self, tb_name, sql_update, sql_where, *para):
        if len(sql_where):
            str_update='UPDATE '+tb_name+' SET '+sql_update+' WHERE '+sql_where
        else:
            str_update='UPDATE '+tb_name+' SET '+sql_update
            
        for i in range(len(para)):
            s=re.search(r"(?<=')"+'##'+str(i)+r"(?=')", str_update)
            str_update=str_update.replace('##'+str(i), str(para[i]))
        return str_update
    
    def DB_Base_Create_SqlCmd_INSERT(self, tb_name, titleNameList, data):
        str_insert=''
        if 'mysql'==global_value.db_type:
            str_insert=self.DB_MySql_Create_SqlCmd_INSERT(tb_name, titleNameList, data)
        elif 'sqlite'==global_value.db_type:
            str_insert=self.DB_Sqlite_Create_SqlCmd_INSERT(tb_name, titleNameList, data)
        else:
            assert 0
        return str_insert

    def DB_Base_Query(self, query):
        #执行查询操作
        self.db_curs.execute(query)

        #取出列名称
        #names=[f[0] for f in self.db_curs.description]
        
        l_temp=[]
        for row in self.db_curs.fetchall():
            l_temp.append(list(copy.deepcopy(row)))
        return l_temp
    
    def DB_Base_Update(self, update):
        try:
            #执行更新操作
            self.db_curs.execute(update)
        except Exception as err:
            print ('Update Fail: ')
            print (unicode(update, 'utf-8'))
            print (err)
            print traceback.format_exc()
            return -1
        return 0
    
    def DB_Base_InsertNewEmptyItem(self, insert):
        try:
            #执行插入操作        
            self.db_curs.execute(insert)
        except Exception as err:
            print ('Insert Fail: ')
            print (unicode(insert, 'utf-8'))
            print (err)
            print traceback.format_exc()
            return -1
        return 0
    

