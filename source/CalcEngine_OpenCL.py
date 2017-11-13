# -*- coding: utf-8 -*-

#Python 2.7.x
import pyopencl as cl
import math
import numpy as np
__metaclass__ = type


#将cl的相关类和函数挪到这里



class OpenCL_WorkUnit():
    platform_handle=''
    device_handle=''
    context_handle=''
    cmdque_handle=''
    
    
    def __init__(self):
        
        return

class OpenCL_Cap():
    #这个提供枚举所有可用OpenCL资源的能力，即枚举所有的设备，每个设备创建一个上下文，创建一个命令队列
    handles=[]
    
    
    def __init__(self):
        self.handles=[]
        
        platforms = self._enum_platform_()
        for item_platform in platforms:
            self._show_platform_(item_platform)
            #对每种平台找出设备
            devs = self._enum_device_(item_platform)
            for item_dev in devs:
                self._show_device_(item_dev)
                for i in range(2):
                    
                    #对item_platform中的item_dev创建上下文和命令队列
                    cont = self._create_context_(item_platform, item_dev)
                    cmdque = self._create_commondQueue_(cont, item_dev)
                    
                    work_handle = OpenCL_WorkUnit()
                    work_handle.platform_handle = item_platform
                    work_handle.device_handle = item_dev
                    work_handle.context_handle = cont
                    work_handle.cmdque_handle = cmdque
                    
                    self.handles.append(work_handle)
                pass
            pass
        return
    
    def _enum_platform_(self):
        platform_list=[]
        for platform in cl.get_platforms():
            platform_list.append(platform)        
        return platform_list
    
    def _show_platform_(self, pf):
        print '----platform----'
        print 'NAME:',pf.get_info(cl.platform_info.NAME)
        print 'EXTENSIONS:',pf.get_info(cl.platform_info.EXTENSIONS)
        return
    
    def _enum_device_(self, platform):
        device=[]        
        for found_device in platform.get_devices():
            device.append(found_device)
        return device
    
    def _show_device_(self, dev):
        print '\t----device----'
        print '\tNAME:',dev.get_info(cl.device_info.NAME)
        print '\tTYPE:',cl.device_type.to_string(dev.type) 
        print '\tDRIVER_VERSION:',dev.get_info(cl.device_info.DRIVER_VERSION)
        print '\tGLOBAL_MEM_SIZE:',dev.get_info(cl.device_info.GLOBAL_MEM_SIZE)
        print '\tLOCAL_MEM_SIZE:',dev.get_info(cl.device_info.LOCAL_MEM_SIZE)
        print '\tMAX_COMPUTE_UNITS:',dev.get_info(cl.device_info.MAX_COMPUTE_UNITS)
        print '\tMAX_WORK_GROUP_SIZE:',dev.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
        print '\tMAX_WORK_ITEM_DIMENSIONS:',dev.get_info(cl.device_info.MAX_WORK_ITEM_DIMENSIONS)
        print '\tMAX_WORK_ITEM_SIZES:',dev.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)
        print '\tOPENCL_C_VERSION:',dev.get_info(cl.device_info.OPENCL_C_VERSION)
        print '\tVENDOR:',dev.get_info(cl.device_info.VENDOR)
        return
    
    def _get_device_info_(self, dev):
        res={}
        res['NAME'] = dev.get_info(cl.device_info.NAME)
        res['TYPE'] = cl.device_type.to_string(dev.type) 
        res['DRIVER_VERSION'] = dev.get_info(cl.device_info.DRIVER_VERSION)
        res['GLOBAL_MEM_SIZE'] = dev.get_info(cl.device_info.GLOBAL_MEM_SIZE)
        res['LOCAL_MEM_SIZE'] = dev.get_info(cl.device_info.LOCAL_MEM_SIZE)
        res['MAX_COMPUTE_UNITS'] = dev.get_info(cl.device_info.MAX_COMPUTE_UNITS)
        res['MAX_WORK_GROUP_SIZE'] = dev.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
        res['MAX_WORK_ITEM_DIMENSIONS'] = dev.get_info(cl.device_info.MAX_WORK_ITEM_DIMENSIONS)
        res['MAX_WORK_ITEM_SIZES'] = dev.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)
        res['OPENCL_C_VERSION'] = dev.get_info(cl.device_info.OPENCL_C_VERSION)
        res['VENDOR'] = dev.get_info(cl.device_info.VENDOR)
        
        return
        
    def _create_context_(self, platform, device):
        '''创建上下文'''
        ctx_props = cl.context_properties
        props = []
        props.append((ctx_props.PLATFORM, platform))
        Context=cl.Context(devices=[device], properties=props)
        return Context
    
    def _create_commondQueue_(self, context, device):
        '''创建命令队列群'''
        CmdQue = cl.CommandQueue(context, device, cl.command_queue_properties.PROFILING_ENABLE)
        return CmdQue
      


class OpenCL_Element():
    chosen_platform_handle=''
    chosen_platform_name=''
    chosen_platform_extensions=''
    chosen_devices=[]
    myContext=''
    myCmdQue=[]
    myProgram=''
    myKernel=''
    
    def __init__(self):
        '''从配置文件中读取以前所选择的平台，如果以前没有选择过，或者以前选择的平台不在当前PC所拥有的平台列表中，就重新选择平台'''
        
        self.chosen_platform_handle
        return
    
    def __select_platforms__(self):
        platform_list=[]
        for platform in cl.get_platforms():
            platform_list.append(platform)
            
        for plat_index,platform in enumerate(platform_list):
            print "[%d] : %s, %s" % (plat_index, platform.get_info(cl.platform_info.NAME), platform.get_info(cl.platform_info.EXTENSIONS))
        
        chosen_index='0'
        while True:
            index_list=[str(item) for item in range(len(platform_list))]
            chosen_index=raw_input('which platform should be used?(%s)' % '/'.join(index_list))
            if chosen_index in index_list:
                break
        
        chosen_index=int(chosen_index)
        
        self.chosen_platform_handle = platform_list[chosen_index]
        self.chosen_platform_name=platform_list[chosen_index].get_info(cl.platform_info.NAME)
        self.chosen_platform_extensions=platform_list[chosen_index].get_info(cl.platform_info.EXTENSIONS)
        
        return
        
    def __select_devices__(self):
        '''选择要使用的device'''
        
        assert self.chosen_platform_handle
        
        device=[]
        
        #优先选择GPU
        for found_device in self.chosen_platform_handle.get_devices():
            if cl.device_type.to_string(found_device.type) == 'GPU':
                device.append(found_device)
        #如果没有GPU就选择CPU
        if not device:
            for found_device in self.chosen_platform_handle.get_devices():
                if cl.device_type.to_string(found_device.type) == 'CPU':
                    device.append(found_device)
        
        assert device
        
        for dev_index,dev in enumerate(device):
            print '----DEV ',dev_index,'----'
            print 'NAME:',dev.get_info(cl.device_info.NAME)
            print 'DRIVER_VERSION:',dev.get_info(cl.device_info.DRIVER_VERSION)
            print 'GLOBAL_MEM_SIZE:',dev.get_info(cl.device_info.GLOBAL_MEM_SIZE)
            print 'LOCAL_MEM_SIZE:',dev.get_info(cl.device_info.LOCAL_MEM_SIZE)
            print 'MAX_COMPUTE_UNITS:',dev.get_info(cl.device_info.MAX_COMPUTE_UNITS)
            print 'MAX_WORK_GROUP_SIZE:',dev.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
            print 'MAX_WORK_ITEM_DIMENSIONS:',dev.get_info(cl.device_info.MAX_WORK_ITEM_DIMENSIONS)
            print 'MAX_WORK_ITEM_SIZES:',dev.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)
            print 'OPENCL_C_VERSION:',dev.get_info(cl.device_info.OPENCL_C_VERSION)
            print 'VENDOR:',dev.get_info(cl.device_info.VENDOR)
        
        self.chosen_devices=device  #一个或多个device组成的列表
        
        return
        
    def __create_context__(self):
        '''创建上下文'''
        ctx_props = cl.context_properties
        props = []
        props.append((ctx_props.PLATFORM, self.chosen_platform_handle))
        self.myContext=cl.Context(devices=self.chosen_devices, properties=props)
        
    def __create_commondQueue__(self):
        '''创建命令队列群，给每个设备创建一个命令队列，所有的命令队列放入一个列表来管理'''
        self.myCmdQue=map(lambda x: cl.CommandQueue(self.myContext, x, cl.command_queue_properties.PROFILING_ENABLE), self.chosen_devices)
        assert self.myCmdQue
        
    def __getProgram__(self, filename):
        program_file = open(filename, 'r')
        program_text = program_file.read()
        program_file.close()
        self.myProgram = cl.Program(self.myContext, program_text)
        return
    
    def __buildProgram__(self):
        self.myKernel = self.myProgram.build()
        log=self.myProgram.get_build_info(self.chosen_devices[0], cl.program_build_info.LOG)
        print "Build Log:"
        print log
        return
    
    def getProgramKernel(self, filename):
        self.__getProgram__(filename)
        self.__buildProgram__()
        return
    
    def create_context_commondqueue(self):
        self.__select_platforms__()
        self.__select_devices__()
        self.__create_context__()
        self.__create_commondQueue__()
    
            
    


class opencl_algorithms_engine():
    cl_oe=''
    cl_ctx=''
    cl_queue=''
    cl_device=''
    cl_deviceInfo_maxGroupSize=0
    
    def __init__(self, device_index=0, kernelFile=''):
        self.cl_oe=OpenCL_Element()
        self.cl_oe.create_context_commondqueue()
        self.cl_oe.getProgramKernel(kernelFile)
    
        self.cl_ctx = self.cl_oe.myContext
        self.cl_queue = self.cl_oe.myCmdQue[device_index]
        self.cl_device = self.cl_oe.chosen_devices[device_index]
    
        self.cl_deviceInfo_maxGroupSize = self.cl_device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
        self.cl_deviceInfo_maxGroupSize = int(math.pow(2, math.floor(math.log(self.cl_deviceInfo_maxGroupSize,2))))
        return
    
    def delete_engine(self):
        assert 0
        return
    
    def cl_algorithm_showClMemStruct(self):
        rows=9
        cols=4
        node_info_block_size=64
        ret_size = node_info_block_size*rows*cols

        ###################################
        dataSet=(10*np.random.random_sample((rows,cols))).astype(np.int32)
        print '--------'
        print "raw dataSet:",dataSet
        
        a_g = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=dataSet)
        res_g = cl.Buffer(self.cl_ctx, cl.mem_flags.WRITE_ONLY, ret_size*np.dtype('int32').itemsize)
        global_size=dataSet.shape
        local_size=(global_size[0]/3, global_size[1]/2)
        self.cl_oe.myKernel.showClMemStruct_2dim(self.cl_queue, global_size, local_size, a_g, res_g)
        res_np = np.zeros(ret_size).astype(np.int32)
        cl.enqueue_copy(self.cl_queue, res_np, res_g)
        
        print "global_size:",global_size
        print "local_size:",local_size
        
        for row in range(rows):
            for col in range(cols):
                res_np_temp = res_np[(row*cols+col)*node_info_block_size:]
                print "gid0:",res_np_temp[0]
                print "gid1:",res_np_temp[1]
                print "gszie0:",res_np_temp[2]
                print "gszie1:",res_np_temp[3]
                print "goffset0:",res_np_temp[4]
                print "goffset1:",res_np_temp[5]
                print "grpnum0:",res_np_temp[6]
                print "grpnum1:",res_np_temp[7]
                print "grpid0:",res_np_temp[8]
                print "grpid1:",res_np_temp[9]
                print "lid0:",res_np_temp[10]
                print "lid1:",res_np_temp[11]
                print "lszie0:",res_np_temp[12]
                print "lszie1:",res_np_temp[13]
                
                print "data:",res_np_temp[14]
                pass
            pass
        
        ###################################
        dataSet=(10*np.random.random_sample((rows,cols))).astype(np.int32)
        print '--------'
        print "raw dataSet:",dataSet
        
        a_g = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=dataSet)
        res_g = cl.Buffer(self.cl_ctx, cl.mem_flags.WRITE_ONLY, ret_size*np.dtype('int32').itemsize)
        global_size=dataSet.shape
        local_size=None
        self.cl_oe.myKernel.showClMemStruct_2dim(self.cl_queue, global_size, local_size, a_g, res_g)
        res_np = np.zeros(ret_size).astype(np.int32)
        cl.enqueue_copy(self.cl_queue, res_np, res_g)
                
        print "global_size:",global_size
        print "local_size:",local_size
                
        for row in range(rows):
            for col in range(cols):
                res_np_temp = res_np[(row*cols+col)*node_info_block_size:]
                print "gid0:",res_np_temp[0]
                print "gid1:",res_np_temp[1]
                print "gszie0:",res_np_temp[2]
                print "gszie1:",res_np_temp[3]
                print "goffset0:",res_np_temp[4]
                print "goffset1:",res_np_temp[5]
                print "grpnum0:",res_np_temp[6]
                print "grpnum1:",res_np_temp[7]
                print "grpid0:",res_np_temp[8]
                print "grpid1:",res_np_temp[9]
                print "lid0:",res_np_temp[10]
                print "lid1:",res_np_temp[11]
                print "lszie0:",res_np_temp[12]
                print "lszie1:",res_np_temp[13]
                        
                print "data:",res_np_temp[14]
                pass
            pass
        
        
        ###################################
        elementNum=12
        ret_size = node_info_block_size*elementNum
        
        
        dataSet=(10*np.random.random_sample(elementNum)).astype(np.int32)
        print '--------'
        print "raw dataSet:",dataSet
        a_g = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=dataSet)
        res_g = cl.Buffer(self.cl_ctx, cl.mem_flags.WRITE_ONLY, ret_size*np.dtype('int32').itemsize)
        global_size=(elementNum, )
        local_size=(elementNum/3, )
        self.cl_oe.myKernel.showClMemStruct_1dim(self.cl_queue, global_size, local_size, a_g, res_g)
        res_np = np.zeros(ret_size).astype(np.int32)
        cl.enqueue_copy(self.cl_queue, res_np, res_g)
        
        print "global_size:",global_size
        print "local_size:",local_size        
        
        for index in range(elementNum):
            res_np_temp = res_np[index*node_info_block_size:]
            print "gid0:",res_np_temp[0]
            print "gszie0:",res_np_temp[1]
            print "goffset0:",res_np_temp[2]
            print "grpnum0:",res_np_temp[3]
            print "grpid0:",res_np_temp[4]
            print "lid0:",res_np_temp[5]
            print "lszie0:",res_np_temp[6]
        
            print "data:",res_np_temp[7]
            pass
        
        ###################################
        dataSet=(10*np.random.random_sample(elementNum)).astype(np.int32)
        print '--------'
        print "raw dataSet:",dataSet
        a_g = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=dataSet)
        res_g = cl.Buffer(self.cl_ctx, cl.mem_flags.WRITE_ONLY, ret_size*np.dtype('int32').itemsize)
        global_size=(elementNum, )
        local_size=None
        self.cl_oe.myKernel.showClMemStruct_1dim(self.cl_queue, global_size, local_size, a_g, res_g)
        res_np = np.zeros(ret_size).astype(np.int32)
        cl.enqueue_copy(self.cl_queue, res_np, res_g)
    
        print "global_size:",global_size
        print "local_size:",local_size        
    
        for index in range(elementNum):
            res_np_temp = res_np[index*node_info_block_size:]
            print "gid0:",res_np_temp[0]
            print "gszie0:",res_np_temp[1]
            print "goffset0:",res_np_temp[2]
            print "grpnum0:",res_np_temp[3]
            print "grpid0:",res_np_temp[4]
            print "lid0:",res_np_temp[5]
            print "lszie0:",res_np_temp[6]
    
            print "data:",res_np_temp[7]
            pass        
        
              
        return
    
    def unitTest_common(self, cpu_engine, inputSize, outType, func):
        #构造测试数据
        inPara = [(10*np.random.random_sample(item)).astype(np.fload32) for item in inputSize]
        
        ret_cpu = getattr(cpu_engine, func)(tuple(inPara))
        ret_ocl = getattr(self, func)(tuple(inPara))

        if outType==0:  #<type 'numpy.ndarray'>
            if ((c_np_cpu - c_np_ocl).max()) < 0.00001:
                return True
        if outType==1:  #type is number like int, float
            if (c_np_cpu - c_np_ocl)<0.00001:
                return True
        if outType==2:  #type is tuple
            if ((np.array(c_np_cpu) - np.array(c_np_ocl)).max()) < 0.00001:
                return True
    
        return False      
    
    #实现各种算法
    #求numpy一维数据中的最大值和最小值    
    def algorithm_vector_max_min(self, a_np):
        
        ceil_len=int(math.pow(2, int(math.ceil(math.log(a_np.shape[0],2)))))   #大于等于a_np长度的2的正数次幂的最小值
        if ceil_len != a_np.shape[0]:   #如果a_np长度不是恰好是某个2的正数次幂
            a_np = np.tile(a_np,2)      #将a_np重复一次
            a_np = a_np[:ceil_len]      #截取2的整数次幂个数据
        
        a_g = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a_np)
        max_g = cl.Buffer(self.cl_ctx, cl.mem_flags.WRITE_ONLY, np.dtype('float32').itemsize)
        min_g = cl.Buffer(self.cl_ctx, cl.mem_flags.WRITE_ONLY, np.dtype('float32').itemsize)
        
        global_size=a_np.shape[0]/4     #由于数据以float4进行计算，所以全部数据按float4的个数计算，是原始数据长度的1/4
        local_size=self.cl_deviceInfo_maxGroupSize
        #第二个参数是全局需要的工作项的数量，要以元组形式给出，元组数据的数量依据数据划分的维度而定
        #第三个参数是每个工作组的工作项的数量，也同样是以元组形式给出
        #'''归并一次'''
        if global_size<local_size:
            local_size = global_size
        self.cl_oe.myKernel.vector_max_min(self.cl_queue, (global_size, ), (local_size, ), a_g, cl.LocalMemory(local_size*4*np.dtype('float32').itemsize), cl.LocalMemory(local_size*4*np.dtype('float32').itemsize))
        
        #调试代码。确认一下中间结果
        #temp_a_np = np.empty_like(a_np)
        #cl.enqueue_copy(self.cl_queue, temp_a_np, a_g)
        #print temp_a_np[:2*global_size/local_size]
        
        
        #'''继续归并'''
        while (2*global_size/local_size) > local_size:
            #在上一次归并中，使用了(global_size/local_size)个工作组，会产生(global_size/local_size)*2个float4数据，
            #如果(global_size/local_size)*2还大于一个工作组中最大工作项数量(即local_size)，就需要继续归并
            global_size = 2*global_size/local_size
            self.cl_oe.myKernel.vector_max_min(self.cl_queue, (global_size, ), (local_size, ), a_g, cl.LocalMemory(local_size*np.dtype('float32').itemsize), cl.LocalMemory(local_size*np.dtype('float32').itemsize))
        
        #现在剩余的float4数据数量已经低于或等于一个工作组的计算容量
        global_size = 2*global_size/local_size
        self.cl_oe.myKernel.vector_max_min_complete(self.cl_queue, (global_size, ), None, a_g, cl.LocalMemory(local_size*4*np.dtype('float32').itemsize), cl.LocalMemory(local_size*4*np.dtype('float32').itemsize), max_g, min_g)
        
        max_np=np.zeros(1).astype(np.float32)
        min_np=np.zeros(1).astype(np.float32)
        cl.enqueue_copy(self.cl_queue, max_np, max_g)
        cl.enqueue_copy(self.cl_queue, min_np, min_g)       
        
        return max_np[0], min_np[0]

    
    def unitTest_algorithm_vector_max_min(self, cpu_engine, func):
        return self.unitTest_common(cpu_engine, [(10000,)], 2, func)
    
    #实现矩阵的复制
    def algorithm_matrix_copy(self, a_np):
        a_g = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a_np)
        b_g = cl.Buffer(self.cl_ctx, cl.mem_flags.WRITE_ONLY, a_np.nbytes)
        self.cl_oe.myKernel.copy_matrix(self.cl_queue, (a_np.shape[0], a_np.shape[1]/4), None, a_g, b_g)
        b_np = np.empty_like(a_np)
        cl.enqueue_copy(self.cl_queue, b_np, b_g)
        return b_np
    
    def unitTest_algorithm_matrix_copy(self, cpu_engine, func):
        return self.unitTest_common(cpu_engine, [(10000,5000)], 0, func)
    
    #矩阵的每一行减去一个向量
    def algorithm_matrix_vector_sub(self, a_np, b_np):
        a_g = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a_np)
        b_g = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b_np)
        self.cl_oe.myKernel.matrix_vector_sub(self.cl_queue, (a_np.shape[0], a_np.shape[1]/4), None, a_g, b_g)
        cl.enqueue_copy(self.cl_queue, a_np, a_g)
        return a_np
    
    def unitTest_algorithm_matrix_vector_sub(self, cpu_engine, func):
        return self.unitTest_common(cpu_engine, [(10000,5000), (5000,)], 0, func)
    
    #求curve曲线的第一个curveLen长度子曲线与后继一定偏移的curveLen长度子曲线的距离，一共要计算0-(calcLen-1)的偏移范围的子曲线之间距离值。
    #比如[a,b,c,d,e,f,g,h,i,j,k,l]-->[dist((a,b,c),(b,c,d)), dist((a,b,c),(c,d,e)), dist((a,b,c),(d,e,f)), ...]
    def algorithm_curve_distance(curve, curveLen, calcLen):
        #为了优化运算速度，所以计算的时候要优化算法
        #先将输入的曲线[a,b,c,d,e,f,g,h,i,j,k,l,...],转换为
        #    [
        #        [a,b,c],
        #        [b,c,d],
        #        [c,d,e],
        #        [d,e,f],
        #        [e,f,g],
        #        [f,g,h],
        #        ...
        #    ]
        #然后就只要在列之间进行计算
        curve_np = np.array([curve[i:]+i*[0.0] for i in range(curveLen)]).T
        curve_g = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=curve_np)
                
        ret_g = cl.Buffer(self.cl_ctx, cl.mem_flags.WRITE_ONLY, len(curve)*np.dtype('float32').itemsize)
        
        self.cl_oe.myKernel.vector_curve_distance(self.cl_queue, (curve_np.shape[0], ), None, curve_g, cl.LocalMemory(curveLen*np.dtype('float32').itemsize), ret_g)
        ret_np = np.zeros(len(curve))
        cl.enqueue_copy(self.cl_queue, ret_np, ret_g)        
        return ret_np[:calcLen]
    
    def algorithm_matrix_vector_curve_distance(self, dataSet, daysLenList, calcLen):
        distance_array=[]
        for col_index in range(dataSet.shape[1]):
            curve=dataSet[:,col_index]
            daysLen=daysLenList[col_index]
            temp_dist = sekf.algorithm_curve_distance(curve, daysLen, calcLen)
            distance_array.append(temp_dist)
        return np.array(distance_array).T
    
    #矩阵的每一行除一个向量
    def algorithm_matrix_vector_div(self, a_np, b_np):
        a_g = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a_np)
        b_g = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b_np)
        self.cl_oe.myKernel.matrix_vector_div(self.cl_queue, a_np.shape, None, a_g, b_g)
        cl.enqueue_copy(self.cl_queue, a_np, a_g)
        return a_np
    
    def unitTest_algorithm_matrix_vector_div(self, cpu_engine, func):
        return self.unitTest_common(cpu_engine, [(10000,5000), (5000,)], 0, func)
    
    def algorithm_matrix_mul_k_float(self, a_np, b_np):
        a_g = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=a_np)
        k_g = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b_np)
        #输入数据的矩阵是二维的。
        self.cl_oe.myKernel.matrix_mul_k_float(self.cl_queue, (a_np.size, ), (a_np.shape[1], ), a_g, k_g)
        normData = np.empty_like(a_np)
        cl.enqueue_copy(self.cl_queue, normData, a_g)
        
        return normData
    
    def unitTest_algorithm_matrix_mul_k_float(self, cpu_engine, func):
        return self.unitTest_common(cpu_engine, [(10000,5000), (5000,)], 0, func)
    
    def algorithm_matrix_element_square_float(self, a_np):
        a_g = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a_np)
        self.cl_oe.myKernel.matrix_element_square_float(self.cl_queue, a_np.shape, None, a_g)
        sqDiffMat=np.empty_like(a_np)
        cl.enqueue_copy(self.cl_queue, sqDiffMat, a_g)        
        return sqDiffMat
    
    def unitTest_algorithm_matrix_element_square_float(self, cpu_engine, func):
        return self.unitTest_common(cpu_engine, [(10000,5000)], 0, func)
    
    def algorithm_matrix_rowadd_rooting(self, a_np):
        a_g = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a_np)
        b_g = cl.Buffer(self.cl_ctx, cl.mem_flags.WRITE_ONLY, a_np.shape[0]*np.dtype('float32').itemsize)
        self.cl_oe.myKernel.matrix_rowadd_rooting(self.cl_queue, (a_np.size, ), (a_np.shape[1], ), a_g, cl.LocalMemory(a_np.shape[1]*np.dtype('float32').itemsize), b_g)
        ret = np.zeros(a_np.shape[0]).astype(np.float32)
        cl.enqueue_copy(self.cl_queue, ret, b_g)
        return ret
    
    def unitTest_algorithm_matrix_rowadd_rooting(self, cpu_engine, func):
        return self.unitTest_common(cpu_engine, [(10000,5000)], 0, func)
    
    #Returns the indices that would sort an array.
    #使用双调算法进行升序排序，返回排序索引
    #a_g是一个一维数组，并且长度是2的幂次
    def algorithm_argsort(self, input_np):
        old_len=len(input_np)
        new_len=int(math.pow(2, int(math.ceil(math.log(len(input_np),2)))))
        a_np = input_np[:]
        max_temp, min_temp = self.algorithm_vector_max_min(a_np)
        a_np = np.concatenate((a_np, np.array([min_temp-1]*(new_len-old_len)).astype(np.float32))) #使用比最小值还小1的值补齐数据长度到2的幂次

        a_g = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=a_np)
        b_g = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.linspace(0, a_np.shape[0]-1, a_np.shape[0]).astype('uint32'))
        
        local_size=self.cl_deviceInfo_maxGroupSize
        global_size = a_np.shape[0]/8;
        if global_size < local_size:
            local_size = global_size
        
        #首个轮次
        self.cl_oe.myKernel.bsort_init(self.cl_queue, (global_size, ), (local_size, ), a_g, cl.LocalMemory(8*local_size*np.dtype('float32').itemsize), b_g, cl.LocalMemory(8*local_size*np.dtype('uint32').itemsize))
        #test_value=np.empty_like(a_np)
        #cl.enqueue_copy(self.cl_queue, test_value, a_g)

        #阶段n的组合和排序
        num_stages = (int)(global_size/local_size)
        high_stage = 2
        while high_stage < num_stages:
            high_stage_g = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([high_stage]))
            stage = high_stage
            while stage > 1:
                stage_g = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([stage]))
                self.cl_oe.myKernel.bsort_stage_n(self.cl_queue, (global_size, ), (local_size, ), a_g, b_g, stage_g, high_stage_g)
                
                #over proc of "while stage > 1:"
                stage >>= 1
                pass
            
            self.cl_oe.myKernel.bsort_stage_0(self.cl_queue, (global_size, ), (local_size, ), a_g, cl.LocalMemory(8*local_size*np.dtype('float32').itemsize), b_g, cl.LocalMemory(8*local_size*np.dtype('uint32').itemsize), high_stage_g)          

            #over proc of "while high_stage < num_stages:"
            high_stage <<= 1
            pass
        
        #最后的组合和排序
        direction=0
        direction_g = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([direction]))
        stage = num_stages
        while stage > 1:
            stage_g = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([stage]))
            self.cl_oe.myKernel.bsort_merge(self.cl_queue, (global_size, ), (local_size, ), a_g, b_g, stage_g, direction_g)
            
            #over proc of "while stage > 1:"
            stage >>= 1
            pass
        self.cl_oe.myKernel.bsort_merge_last(self.cl_queue, (global_size, ), (local_size, ), a_g, cl.LocalMemory(8*local_size*np.dtype('float32').itemsize), b_g, cl.LocalMemory(8*local_size*np.dtype('uint32').itemsize), direction_g)
    
        #读回b_g
        ret = np.zeros(a_np.shape[0]).astype(np.int)
        cl.enqueue_copy(self.cl_queue, ret, b_g)

        return ret[new_len-old_len:]

    def unitTest_algorithm_argsort(self, cpu_engine, func):
        return self.unitTest_common(cpu_engine, [(10000,)], 0, func)



