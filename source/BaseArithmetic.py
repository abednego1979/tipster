# -*- coding: utf-8 -*-

#Python 2.7.x

#V0.01
import numpy as np
import global_value
from BaseFunc import BaseFunc

__metaclass__ = type


class BaseArithmetic(BaseFunc):
    
    #获取一维数据的N均值
    def dataproc_getMean(self, data, meanLen, maxItem=0):
        assert data.ndim==1
        data = np.asarray(data)
        weights = np.ones(meanLen)
        weights /= weights.sum()
        
        a=np.convolve(data, weights, mode='full')[meanLen-1:]
        
        return a
    
    #求价格均线
    def dataproc_getPriceMean(self, colTitles, bData):
        for mean_len in global_value.MEAN_LEN_LIST:
            l=self.dataproc_getMean(bData[:,colTitles.index('AdjClose')], mean_len)#返回的是均值
            lr=l/bData[:,colTitles.index('AdjClose')]
            
            #insert data
            bData.T[colTitles.index('mean_'+str(mean_len))]=l
            bData.T[colTitles.index('mean_'+str(mean_len)+'_RatePrice')]=lr

    
        return bData
    
    def dataproc_getDIFF_12_26(self, colTitles, bData):
        DIFF=bData[:, colTitles.index('mean_12')]-bData[:, colTitles.index('mean_26')]
        DIFF_Rate=DIFF/bData[:,colTitles.index('AdjClose')]#DIFF别价格
        
        #insert
        bData.T[colTitles.index('DIFF_12_26')]=DIFF
        bData.T[colTitles.index('DIFF_12_26_Rate')]=DIFF_Rate
        return bData
    
    def dataproc_getDEA(self, colTitles, bData):
        M=global_value.DEA_M
        DEA=self.dataproc_getMean(bData[:, colTitles.index('DIFF_12_26')], M)
        DEA_Rate=DEA/bData[:,colTitles.index('AdjClose')]#DEA别价格        
        #insert
        bData.T[colTitles.index('DEA_'+str(M))]=DEA
        bData.T[colTitles.index('DEA_'+str(M)+'_Rate')]=DEA_Rate
        return bData
    
    def dataproc_getMACD(self, colTitles, bData):
        M=global_value.DEA_M
    
        MACD=bData[:,colTitles.index('DIFF_12_26')]-bData[:,colTitles.index('DEA_'+str(M))]
        MACD_Rate=MACD/bData[:,colTitles.index('AdjClose')]#MACD别价格

        #insert
        bData.T[colTitles.index('MACD')]=MACD
        bData.T[colTitles.index('MACD_Rate')]=MACD_Rate
        return bData

    def dataproc_getKDJ(self, colTitles, bData):
        N=global_value.KDJ_N
        
        Cn=bData[:,colTitles.index('AdjClose')]  #Cn
        Low=bData[:,colTitles.index('Low')]
        Ln=[]
        for i in range(Low.shape[0]):
            Ln.append(Low[i:i+N].min())
        Ln=np.array(Ln)#Ln
        High=bData[:,colTitles.index('High')]
        Hn=[]
        for i in range(High.shape[0]):
            Hn.append(High[i:i+N].max())
        Hn=np.array(Hn)#Hn
        
        #RSV=(Cn-Ln)/(Hn-Ln)×100    未成熟随机指标值
        with np.errstate(divide='ignore'):
            RSV=((Cn-Ln)/(Hn-Ln))*100.0
        RSV[np.isnan(RSV)] = 0.0
        RSV[np.isinf(RSV)] = 0.0
        
        #K值=2/3×前一日K值+1/3×当日RSV
        #D值=2/3×前一日D值+1/3×当日K值
        #J值=3*当日K值-2*当日D值
        K=[]
        D=[]
        J=[]
        k=50
        d=50
        data_len=bData[:,colTitles.index('AdjClose')].shape[0]
        for i in range(data_len):
            k=k*2/3+RSV[data_len-1-i]/3
            d=d*2/3+k/3
            j=k*3-d*2
            K.append(k)
            D.append(d)
            J.append(j)
        assert len(J)==data_len
        
        #insert
        bData.T[colTitles.index('KDJ_K')]=np.array(K)
        bData.T[colTitles.index('KDJ_D')]=np.array(D)
        bData.T[colTitles.index('KDJ_J')]=np.array(J)
        return bData
    
    def dataproc_getRSI(self, colTitles, bData):
        for N in global_value.RSI_DArray:
            RSI=[]
            PC=bData[:,colTitles.index('PriceIncrease')] #收盘价涨跌幅
            for i in range(PC.shape[0]):
                a=PC[i+1:i+1+N]
                b=np.abs(a)
                A=((a+b)/2).sum()
                B=((b-a)/2).sum()
    
                if A==B==0:
                    RSI.append(0.0)
                else:
                    RSI.append(100.0*A/(A+B))
                    
            #insert
            bData.T[colTitles.index('RSI_'+str(N))]=np.array(RSI)            
        return bData
    
    def dataproc_getBOLL(self, colTitles, bData):
        N=global_value.BOLL_N
        
        MA=self.dataproc_getMean(bData[:,colTitles.index('AdjClose')], N)  #价格N日均线
        #MD=((bData[:,colTitles.index('AdjClose')]-MA)**2).sum()
        MD=(bData[:,colTitles.index('AdjClose')]-MA)**2
        MD=self.dataproc_getMean(MD, N)
        MD=np.sqrt(MD)
        MB=np.append(MA[1:], np.array([0]))
        UP=MB+2*MD
        DN=MB-2*MD
        
        #insert
        bData.T[colTitles.index('BOLL_MA_'+str(N))]=MA
        bData.T[colTitles.index('BOLL_MA_'+str(N)+'_Rate')]=MA/bData[:,colTitles.index('AdjClose')]
        bData.T[colTitles.index('BOLL_UP_'+str(N))]=UP
        bData.T[colTitles.index('BOLL_UP_'+str(N)+'_Rate')]=UP/bData[:,colTitles.index('AdjClose')]
        bData.T[colTitles.index('BOLL_DN_'+str(N))]=DN
        bData.T[colTitles.index('BOLL_DN_'+str(N)+'_Rate')]=DN/bData[:,colTitles.index('AdjClose')]
        return bData
    
    def dataproc_getWR(self, colTitles, bData):
        #WR威廉指标--威廉超买超卖指数,主要用于分析市场短期买卖走势
        #计算方法：n日WMS=(Hn－Ct)/(Hn－Ln)×100,Ct为当天的收盘价；Hn和Ln是n日内（包括当天）出现的最高价和最低价
        for N in global_value.WR_DArray:
            WR=[]
            AC=bData[:,colTitles.index('Close')] #收盘价
            for i in range(AC.shape[0]):
                #过去N天的最高价
                Hn = AC[i:i+N].max()
                #过去N天的最低价
                Ln = AC[i:i+N].min()
                #当天收盘价
                Ct = AC[i]
                
                assert Hn>=Ct,'ERROR:'+str(Hn)+' '+str(Ct)
                assert Ln<=Ct,'ERROR:'+str(Ln)+' '+str(Ct)

                if Hn-Ln:
                    WMS = 100.0*(Hn-Ct)/(Hn-Ln)
                else:
                    WMS=0.0
                WR.append(WMS)
                
            #insert
            bData.T[colTitles.index('WR_'+str(N))]=np.array(WMS)

        return bData
    
    def dataproc_getDMI(self, colTitles, bData):
        #Get DMI
        
        #TR:=EXPMEMA(MAX(MAX(HIGH-LOW,ABS(HIGH-ref(CLOSE,1))),ABS(ref(CLOSE,1)-LOW)),N);
        #HD :=HIGH-ref(HIGH,1);
        #LD :=ref(LOW,1)-LOW;
        #DMP:=EXPMEMA(IF(HD>0&&HD>LD,HD,0),N);
        #DMM:=EXPMEMA(IF(LD>0&&LD>HD,LD,0),N);
        #PDI:= DMP*100/TR,COLORFFFFFF;
        #MDI:= DMM*100/TR,COLOR00FFFF;
        #ADX:= EXPMEMA(ABS(MDI-PDI)/(MDI+PDI)*100,M),COLOR0000FF,LINETHICK2;
        #ADXR:=EXPMEMA(ADX,M),COLOR00FF00,LINETHICK2;{本文来源: www.cxh99.com }
        #DYNAINFO(9)>0 AND CROSS(ADX,MDI) AND CROSS(ADXR,MDI) AND PDI>MDI;            
        
        for N in global_value.DMI_DArray:
            #上升动向（+DM）            
            cur_High = bData[:,colTitles.index('High')]
            last_High = np.hstack((cur_High[1:], np.zeros(1)))
            P_DM = cur_High-last_High
            P_DM = np.where(P_DM>0.0, P_DM, 0.0)
            
            
            #下降动向（-DM）
            cur_Low = bData[:,colTitles.index('Low')]
            last_Low = np.hstack((cur_Low[1:], np.zeros(1)))
            N_DM = last_Low-cur_Low
            N_DM = np.where(N_DM>0.0, N_DM, 0.0)            
            
            assert P_DM.shape==N_DM.shape
            
            #调整上升动向（+DM）和下降动向（-DM）            
            temp_P=np.where(P_DM>N_DM, P_DM, 0.0)
            temp_N=np.where(N_DM>P_DM, N_DM, 0.0)
            P_DM = self.dataproc_getMean(temp_P, N)
            N_DM = self.dataproc_getMean(temp_N, N)
            
            
            #计算真实波幅（TR）
            cur_Close=bData[:,colTitles.index('Close')]
            last_Close=np.hstack((cur_Close[1:], np.zeros(1)))
            TR1=cur_High-cur_Low
            TR2=cur_High-last_Close
            TR3=cur_Low-last_Close
            TR1 = np.where(TR1>TR2, TR1, TR2)
            TR = np.where(TR1>TR3, TR1, TR3)
            TR = self.dataproc_getMean(TR, N)
            
            #计算方向线DI-上升指标
            P_DI = (P_DM / TR)*100.0
            #计算方向线DI-上升指标
            N_DI = (N_DM / TR)*100.0
            
            #计算动向平均数ADX
            DX = P_DI+N_DI
            DX = np.where(DX>0.0, DX, 1.0)
            DX = np.fabs(100.0*(P_DI-N_DI)/DX)
                
            ADX = self.dataproc_getMean(DX, N)
            
            #计算评估数值ADXR
            ADX_NDay = np.hstack((ADX[N:], np.zeros(N)))
            ADXR = (ADX+ADX_NDay)*0.5
            
            #insert
            bData.T[colTitles.index('DMI_PDI_'+str(N))]=P_DI
            bData.T[colTitles.index('DMI_NDI_'+str(N))]=N_DI
            bData.T[colTitles.index('DMI_ADX_'+str(N))]=ADX
            bData.T[colTitles.index('DMI_ADXR_'+str(N))]=ADXR
            pass
                    
        return bData
 

  