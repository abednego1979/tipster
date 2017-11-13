

__kernel void showClMemStruct_2dim(__global int* data, __global int* res)
{
    int base;
    int gid0=get_global_id(0);
    int gid1=get_global_id(1);
    int gsize0=get_global_size(0);
    int gsize1=get_global_size(1);
    int goffset0=get_global_offset(0);
    int goffset1=get_global_offset(1);
    int grpnum0=get_num_groups(0);
    int grpnum1=get_num_groups(1);
    int grpid0=get_group_id(0);
    int grpid1=get_group_id(1);
    int lid0=get_local_id(0);
    int lid1=get_local_id(1);
    int lszie0=get_local_size(0);
    int lszie1=get_local_size(1);
    
    base=(gid0*gsize1+gid1)*64;
    res[base+ 0]=gid0;
    res[base+ 1]=gid1;
    res[base+ 2]=gsize0;
    res[base+ 3]=gsize1;
    res[base+ 4]=goffset0;
    res[base+ 5]=goffset1;
    res[base+ 6]=grpnum0;
    res[base+ 7]=grpnum1;
    res[base+ 8]=grpid0;
    res[base+ 9]=grpid1;
    res[base+10]=lid0;
    res[base+11]=lid1;
    res[base+12]=lszie0;
    res[base+13]=lszie1;
    
    res[base+14]=data[gid0*gsize1+gid1];
    return;
}


__kernel void showClMemStruct_1dim(__global int* data, __global int* res)
{
    int base;
    int gid0=get_global_id(0);
    int gsize0=get_global_size(0);
    int goffset0=get_global_offset(0);
    int grpnum0=get_num_groups(0);
    int grpid0=get_group_id(0);
    int lid0=get_local_id(0);
    int lszie0=get_local_size(0);
    
    base=gid0*64;
    res[base+ 0]=gid0;
    res[base+ 1]=gsize0;
    res[base+ 2]=goffset0;
    res[base+ 3]=grpnum0;
    res[base+ 4]=grpid0;
    res[base+ 5]=lid0;
    res[base+ 6]=lszie0;
    
    res[base+ 7]=data[gid0];
    return;
}


/*
求序列的最大值和最小值

基本算法：
数据将以float4的形式提供给kernel，长度约几千个，超过了一个工作组中工作项的数量。
在一个工作组内部，假定有256个工作项
则数据为：(d0，d1，d2，d3，...，d255)， 求(max(d0,d128), max(d1,d129), max(d2,d130),...,max(d127,d255)), 以及(min(d0,d128), min(d1,d129), min(d2,d130),...,min(d127,d255))
然后不断的重复的将数据缩短两个float4(一个是max值，一个是min值)
当不同工作组都完成计算以后，如果所得的float4数据仍然多于每工作组中工作项的数量，就重复上面的过程。否则使用一个新的内核计算剩余的结果，并最终得到一个float的结果

比如，如果数据有1048567个，那么float4数据就是1048567/4=262144个，假设一个工作组容纳256个工作项，则
第一轮，创建262144/256=1024个组，每个组产生max和min两个float4的数据，所以这一轮结束以后，数据量剩1024*2=2048个float4数据
第二轮，由于2048大于256，所以还要进行一轮的归并。会创建2048/256=8个组，每个组产生max和min两个float4的数据，所以这一轮结束以后，数据量剩8*2=16个float4数据
最后，由于所剩数据量小于256，所以一个工作组就能计算。在末轮计算算法中，计算得到max和min两个float数据（注意不是float4数据）

*/
__kernel void vector_max_min(__global float4* data, __local float4* partial_max, __local float4* partial_min)
{
    int lid = get_local_id(0);  /*在工作组中的id*/
    int group_size = get_local_size(0);     /*工作组的大小，也就是一个工作组的容纳的并行工作项的数量，这个数值的最优数值是从device info中获取的*/
    
    partial_max[lid] = data[get_global_id(0)];      /*将全局数据复制到局部内存*/
    partial_min[lid] = partial_max[lid];
    barrier(CLK_LOCAL_MEM_FENCE);               /*同步各个工作项复制结束*/
    
    for(int i = group_size/2; i>0; i >>= 1)
    {
        if(lid < i)
        {
            partial_max[lid] = fmax(partial_max[lid], partial_max[lid+i]);
            partial_min[lid] = fmin(partial_min[lid], partial_min[lid+i]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if(lid == 0)
    {
        data[2*get_group_id(0)] = partial_max[0];
        data[2*get_group_id(0)+1] = partial_min[0];
    }

}

__kernel void vector_max_min_complete(__global float4* data, __local float4* partial_max, __local float4* partial_min, __global float* max, __global float* min)
{
    int lid = get_local_id(0);          /*在工作组中的id*/    /**/
    int group_size = get_local_size(0); /*工作组的大小*/
    
    partial_max[lid] = data[get_local_id(0)];      /*将全局数据复制到局部内存*/
    partial_min[lid] = partial_max[lid];
    barrier(CLK_LOCAL_MEM_FENCE);
    

    for(int i = group_size/2; i>0; i >>= 1)
    {
        if(lid < i)
        {
            partial_max[lid] = fmax(partial_max[lid], partial_max[lid+i]);
            partial_min[lid] = fmin(partial_min[lid], partial_min[lid+i]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0)
    {
        float4 f4_temp_max=partial_max[0];
        float4 f4_temp_min=partial_min[0];
        *max = fmax(fmax(f4_temp_max.x, f4_temp_max.y), fmax(f4_temp_max.z, f4_temp_max.w));
        *min = fmin(fmin(f4_temp_min.x, f4_temp_min.y), fmin(f4_temp_min.z, f4_temp_min.w));
    }
}


/*
矩阵复制
*/
__kernel void copy_matrix(__global float4* inMatrix, __global float4* outMatrix)
{
    int gid0=get_global_id(0);/*row*/
    int gid1=get_global_id(1);/*col*/
    int gsize1=get_global_size(1);
    
    outMatrix[gid0*gsize1+gid1] = inMatrix[gid0*gsize1+gid1];
}

/*
矩阵减去向量，即矩阵的每一行减去一个向量

基本算法：
矩阵数据是二维的，在kernel中利用get_global_id(0)和get_global_id(1)可以获取一个数据在矩阵中的位置。
对于输入或者输出的二维矩阵，get_global_id(0)是行编号，get_global_id(1)是列编号，即data[get_global_id(0)][get_global_id(1)]
*/
__kernel void matrix_vector_sub(__global float4* matrix, __global float4* vector)
{
    int gid0=get_global_id(0);/*row*/
    int gid1=get_global_id(1);/*col*/
    int gsize1=get_global_size(1);
    
    matrix[gid0*gsize1+gid1] -= vector[gid1];
}

/*

*/
__kernel void vector_curve_distance(__global float* curve_g, __local float* partial_curve, __global float* ret_g)
{
    int gid0=get_global_id(0);/*row index*/
    int lsize0 = get_local_size(0); /*工作组的大小*/
    
    uint index;
    float fSum;

    for (index=0; index<lsize0; index++)
    {
        partial_curve[index] = curve_g[index];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    if (0 != gid0)
    {
        fSum=0.0;
        for (index=0; index<lsize0; index++)
        {
            fSum+=fabs(partial_curve[index]-curve_g[gid0*lsize0+index]);
        }        
        ret_g[gid0-1] = fSum/lsize0;
    }
    
}

/*
矩阵除以向量，即矩阵的每一行以对应点相除的方式处以一个向量
基本算法：类似于sub_matrix_vector
*/
__kernel void matrix_vector_div(__global float* matrix, __global float* vector)
{
    int gid0=get_global_id(0);/*row*/
    int gid1=get_global_id(1);/*col*/
    int gsize1=get_global_size(1);
    
    matrix[gid0*gsize1+gid1] /= vector[gid1];
    
}




/*
矩阵乘以系数

B=A*k

|aw,bx,cy,dz|    |a,b,c,d|    |w|
|ew,fx,gy,hz|    |e,f,g,h|    |x|
|iw,jx,ky,lz| =  |i,j,k,l| op |y|
|mw,nx,oy,pz|    |m,n,o,p|    |z|

由于输入的矩阵在计算中不会互相加减乘除，所以矩阵数据就不放入局部变量中了。但是由于向量要多次使用，所以可以放入局部变量

具体算法：对于m行n列的矩阵，可以看作一个m*n长度的一维数据，把每n个数据送入一个local单元进行处理，这个处理单元也要复制一份系数到local内存
*/
__kernel void matrix_mul_k_float(__global float* matrix, __global float* vector)
{
    int lid0=get_local_id(0);
    int lsize0=get_local_size(0);
    int grpid0=get_group_id(0);
    
    int temp_index=grpid0*lsize0+lid0;
    
    matrix[temp_index] = matrix[temp_index] * vector[lid0];

    return;
}



/*
矩阵元素平方
A^2
|a,b,c,d|^2 =|a^2, b^2, c^2, d^2|
|e,f,g,h|    |e^2, f^2, g^2, h^2|
|i,j,k,l|    |i^2, j^2, k^2, l^2|
|m,n,o,p|    |m^2, n^2, o^2, p^2|
*/
__kernel void matrix_element_square_float(__global float* matrix)
{
    int gid0=get_global_id(0);/*row*/
    int gid1=get_global_id(1);/*col*/
    int gsize1=get_global_size(1);
    
    matrix[gid0*gsize1+gid1] = pow(matrix[gid0*gsize1+gid1], 2);
}


/*

对矩阵的行求和并对和开平方
|a,b,c,d|      |(a+b+c+d)^0.5|
|e,f,g,h|  op= |(e+f+g+h)^0.5|
|i,j,k,l|      |(i+j+k+l)^0.5|
|m,n,o,p|      |(m+n+o+p)^0.5|
数据是一维传递的，长度是row*col，划分的局部长度为col

*/
__kernel void matrix_rowadd_rooting(__global float* matrix, __local float* partial, __global float* dist)
{
    int lid0=get_local_id(0);
    int lsize0=get_local_size(0);
    int grpid0=get_group_id(0);
    
    partial[lid0] = matrix[grpid0*lsize0+lid0];      /* 将全局数据的某一行复制到局部内存 */
    
    /* 每行求和，即对partial进行求和 */
    if (lid0==0)
    {
        float sum=0.0;
        for (int i=0; i<lsize0; i++)
        {
            sum += partial[i];
        }
        
        dist[grpid0] = sqrt(sum);
    }

}



#define UP 0
#define DOWN 1

#define bsort_conster_swap  ((uint4)(0, 0, 1, 1))
#define bsort_conster_mask1 ((uint4)(1, 0, 3, 2))
#define bsort_conster_mask2 ((uint4)(2, 3, 0, 1))
#define bsort_conster_add1 ((uint4)(0, 0, 2, 2))
#define bsort_conster_add2 ((uint4)(0, 1, 0, 1))
#define bsort_conster_add3 ((uint4)(0, 1, 2, 3))



/* Sort elements in a vector */
#define SORT_VECTOR(input, input_index, dir)                                        \
   comp = abs(input > shuffle(input, bsort_conster_mask1)) ^ dir;                                 \
   input = shuffle(input, comp ^ bsort_conster_swap + bsort_conster_add1);              /*构造双调序列*/         \
   input_index = shuffle(input_index, comp ^ bsort_conster_swap + bsort_conster_add1);                          \
   comp = abs(input > shuffle(input, bsort_conster_mask2)) ^ dir;                                 \
   input = shuffle(input, comp * 2 + bsort_conster_add2);                 /*分裂*/                \
   input_index = shuffle(input_index, comp * 2 + bsort_conster_add2);                             \
   comp = abs(input > shuffle(input, bsort_conster_mask1)) ^ dir;                                 \
   input = shuffle(input, comp + bsort_conster_add1);                     /*半区内排序*/           \
   input_index = shuffle(input_index, comp + bsort_conster_add1);                                 \
   

/* Sort elements between two vectors */
#define SWAP_VECTORS(input1, input2, input_index1, input_index2, dir)       /*要求input1已经升序，input2已经降序*/   \
   temp = input1;                                                                                                   \
   temp_index = input_index1;                                                                                       \
   comp = (abs(input1 > input2) ^ dir) * 4 + bsort_conster_add3;                          /*双调分裂操作一次*/                     \
   input1 = shuffle2(input1, input2, comp);                                                                         \
   input2 = shuffle2(input2, temp, comp);                                                                           \
   input_index1 = shuffle2(input_index1, input_index2, comp);                                                       \
   input_index2 = shuffle2(input_index2, temp_index, comp);                                                         \


/* bitonic sort two vectors */
#define SORT_TWO_VECTOR(input1, input2, input_index1, input_index2, dir)      \
    SWAP_VECTORS(input1, input2, input_index1, input_index2, dir);            \
    SORT_VECTOR(input1, input_index1, dir);                                   \
    SORT_VECTOR(input2, input_index2, dir);                                   \



__kernel void bsort8(__global float4 *data, __global uint4 *data_index, int dir)
{

    float4 input1, input2, temp;
    uint4 input_index1, input_index2, temp_index;
    uint4 comp;

    input1 = data[0];
    input2 = data[1];
    input_index1 = data_index[0];
    input_index2 = data_index[1];

    SORT_VECTOR(input1, input_index1, UP);         /*|                         */
    SORT_VECTOR(input2, input_index2, DOWN);       /*|构造双调序列              */
    
    SORT_TWO_VECTOR(input1, input2, input_index1, input_index2, dir);   /*|对双调序列进行排序    */

    data[0] = input1;
    data[1] = input2;
    data_index[0] = input_index1;
    data_index[1] = input_index2;
}



/*
以下几个函数用于进行双调排序，尤其是数据比较多的时候
当数据很多，假设每个工作组有M个工作项，由于每个工作项可以排序2个float4即8个float数据，所以每个工作组能处理8M个float数据，如果总数据量大于8M个，那么就需要多个工作组参与处理。
所以排序要分几个阶段进行，在最底层阶段中，每个工作组将8M个数据排序（升序或者降序，根据组id而定）。而更上层的阶段，需要将相邻的2或4或8或16或...个组的数据进行组合，
组合的结果是这些组的数据之间有大小关系（如要组合相邻两组时，组0的所有数据小于组1的所有数据（升），组2的所有数据大于组3的所有数据（降）），然后对各组再次排序，
以达到相邻组（2或4或8或16或...个组）数据升序或降序。阶段1是排序相邻两个组，阶段2是排序相邻的4个组，阶段3是排序相邻的8个组，...
完成以上动作以后，会形成一个大双调函数，最后再做一次组合和排序就能完成全部的排序

比如数据需要16个工作组处理
阶段3:	+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
        |          up           |           dn          |
        +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
        
阶段2:	+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
        |    up     |    dn     |    up     |    dn     |
        +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+


阶段1:	+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
        | up  |  dn | up  |  dn | up  |  dn | up  |  dn |
        +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

阶段0:	+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
        |up|dn|up|dn|up|dn|up|dn|up|dn|up|dn|up|dn|up|dn|
        +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

排序过程是
0, 1, 0, 2, 1, 0, 3, 2, 1, 0
-  ----  -------  ----------
阶段0：各个组内升序或降序排序
阶段1：相邻两组组合,保证组与相邻组的数据不会重合（大小分开，如组0数据都小于组1，组2数据都大于组3）
阶段0：各个组内升序或降序排序，这时实现了组0/1升序，组2/3降序，组4/5升序...
阶段2：相邻4组组合，保证2组与相邻2组的数据不会重合（大小分开，如组0/1数据都小于组2/3，组4/5数据都大于组6/7）
阶段1：相邻两组组合,保证组与相邻组的数据不会重合（大小分开，如组0数据都小于组1，组2数据都小于组3, 组4数据都大于组5，组6数据都大于组7,）
阶段0：各个组内升序或降序排序，这时实现了组0/1/2/3升序，组4/5/6/7降序，组8/9/10/11升序...
阶段3：相邻8组组合，保证4组与相邻4组的数据不会重合（大小分开，如组0/1/2/3数据都小于组4/5/6/7，组8/9/10/11数据都大于组12/13/14/15）
阶段2：相邻4组组合，保证2组与相邻2组的数据不会重合（大小分开，如组0/1数据都小于组2/3，组4/5数据都小于组6/7, 组8/9数据都大于组10/11，组12/13数据都小于组14/15）
阶段1：相邻两组组合,保证组与相邻组的数据不会重合（大小分开，如组0数据都小于组1，组2数据都小于组3,..., 组8数据都大于组9, 组10数据都大于组11, ...）
阶段0：各个组内升序或降序排序，这时实现了组0/1/2/3/4/5/6/7升序，组8/9/10/11/12/13/14/15降序

*/

/* Perform initial sort */
/*
在开始排序时，每个工作项都读入两个向量，并对各自所分配到的向量分量进行排序，
然后同一工作组的工作项在对赋给工作组的每一个数据点进行排序
*/
__kernel void bsort_init(__global float4 *g_data, __local float4 *l_data, __global uint4 *index_data, __local uint4 *l_index_data)
{
    float4 input1, input2, temp;
    uint4 comp;
    uint id, dir, global_start, size, stride;
    uint4 input_index1, input_index2, temp_index;
    
    uint lid0,gid0,lsize0;
    
    lid0=get_local_id(0);
    gid0=get_group_id(0);
    lsize0=get_local_size(0);
    
    id = lid0 * 2;                                       /*|                 */
    global_start = gid0 * lsize0 * 2 + id;    /*|  查找全局地址    */
    
    input1 = g_data[global_start];              /*|                             */
    input2 = g_data[global_start+1];            /*|  从这里取两个float4向量      */
    input_index1 = index_data[global_start];
    input_index2 = index_data[global_start+1];    
    
    SORT_VECTOR(input1, input_index1, UP);                    /*|  对第一个向量升序排列      */
    SORT_VECTOR(input2, input_index2, DOWN);                  /*|  对第二个向量降序排列      */

    /* Swap corresponding elements of input 1 and 2 */
    dir = lid0 % 2;
    SORT_TWO_VECTOR(input1, input2, input_index1, input_index2, dir);       /*|对双调序列进行排序     */
    
    l_data[id] = input1;
    l_data[id+1] = input2;
    l_index_data[id] = input_index1;
    l_index_data[id+1] = input_index2;
    

    /*运行到这里，单个工作项负责的两个float4向量已经完成排序，方向与数据位置有关，0/2/4/...工作组的数据升序，其余降序*/
    
    /*########下面开始对工作组内的所有数据进行排序*/
    /*1.构造组内双调序列/Create bitonic set*/
    /*
    这部分代码将组内的数据排布为一个双调序列
    代码比较难懂，不过还是必须要搞懂
    可以尝试举例说明
    
    假设一个工作组有16个工作项，那么数据就是32个float4，并且相邻的两个float4已经同升序或同降序排列
    即0,1,位置的float4是升序，2,3位置是降序，4,5位置float4是升序，6,7位置的float4是降序...
    
    ########
    在循环中，首次循环size=2，stride的循环只有1次，即stride=size=2，先看看size=2时的处理过程
    
    local_id = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15
        
          id = 0, 1, 4, 5, 8, 9, 12,13,16,17,20,21,24,25,28,29
   id+stride = 2, 3, 6, 7, 10,11,14,15,18,19,22,23,26,27,30,31
         dir = u, u, d, d, u, u, d, d, u, u, d, d, u, u, d, d,
    
    然后在stride循环中做了一次向量分裂，在(0,2),(1,3),(4,6),(5,7),...中
    双调分裂的结果是数据0放的是原来数据0和数据2的较小值，数据1放的是原数据1和数据3的较小值，数据2放的是原数据0和数据2的较大值，数据3放的是原数据1和数据3的较大值
    由于之前已经排过序了，所以原数据1肯定大于原数据0，原数据3肯定小于原数据2，所以刚才双调分裂的结果，一定还是数据0和数据1全小于数据2和数据3。
    
    循环结束后（size=2时，循环只有一次），各工作项对各自负责的数据进行排序注意id有变化。
    比如第0号工作项，负责排序数据0和数据1，由于数据0小于数据1，所以双调重排的分裂动作没有实际效果，然后是对数据进行排序，结果是数据0和数据1升序，数据2和数据3也升序，由于数据0和数据1中最大的数据是原来数据中较小的部分，其最大的数也比现在数据2和数据3中的最小值还小，所以现在数据0,1,2,3这4个float4数据是升序的。
    所以size=2这个循环结束时，(0-3)升序，(4,7)降序，(8,11)升序，(12,15)降序...
    当size=4这个循环结束时，(0-7)升序，(8,15)降序，(16,23)升序，(24,31)降序
    当size=8这个循环结束时，(0-15)升序，(16,31)降序，这个工作组负责的数据就变成了一个大的双调序列
    */
    
    for(size = 2; size < lsize0; size <<= 1)     /*对上半部分进行操作*/
    {
        dir = lid0/size & 1;
        
        for(stride = size; stride > 1; stride >>= 1)                                                                    /*|                     */
        {                                                                                                               /*|                     */
            barrier(CLK_LOCAL_MEM_FENCE);                                                                               /*| 对下半部分进行操作   */
            id = lid0 + (lid0/stride)*stride;                                                     /*|                     */
            SWAP_VECTORS(l_data[id], l_data[id + stride], l_index_data[id], l_index_data[id + stride], dir);            /*|                     */
        }                                                                                                               /*|                     */
        barrier(CLK_LOCAL_MEM_FENCE);
        
        id = lid0 * 2;
        input1 = l_data[id];
        input2 = l_data[id+1];
        input_index1 = l_index_data[id];
        input_index2 = l_index_data[id+1];
        SORT_TWO_VECTOR(input1, input2, input_index1, input_index2, dir);           /*|对双调序列进行排序     */    
        l_data[id] = input1;
        l_data[id+1] = input2;
        l_index_data[id] = input_index1;
        l_index_data[id+1] = input_index2;
    }
    

    
    /*下面对这个工作组的负责的这个大双调序列进行升序或降序排列*/
    
    /* Perform bitonic merge */
    dir = gid0 % 2;      /*偶数工作组处理的数据要升序，奇数工作组的要降序*/
    
    for(stride = lsize0; stride > 1; stride >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        id = lid0 + (lid0/stride)*stride;
        SWAP_VECTORS(l_data[id], l_data[id + stride], l_index_data[id], l_index_data[id + stride], dir)
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    

 
    /* Perform final sort */
    id = lid0 * 2;
    input1 = l_data[id];
    input2 = l_data[id+1];
    input_index1 = l_index_data[id];
    input_index2 = l_index_data[id+1];
    SORT_TWO_VECTOR(input1, input2, input_index1, input_index2, dir);           /*|对双调序列进行排序     */    
    g_data[global_start] = input1;
    g_data[global_start+1] = input2;
    index_data[global_start] = input_index1;
    index_data[global_start+1] = input_index2;
}

/* Perform successive stages of the bitonic sort */
/*
执行逐个阶段(除了阶段0)的分裂操作

参数:
float4 *g_data----数据
uint high_stage----对应前面算法介绍中的阶段数，2代表阶段1，4代表阶段2，8代表阶段3
uint stage:----子状态，取值为2的幂。最大等于high_stage

比如high_stage=2，则stage取值只能是2，这时执行阶段1的动作。即对原组0和组1的数据升序，原组2和组3的数据降序，原组4和组5的数据升序，...（原偶数组已经升序，奇数组已经降序）

*/
__kernel void bsort_stage_n(__global float4 *g_data, __global uint4 *index_data, __global uint *stage, __global int *high_stage)
{

    float4 input1, input2, temp;
    uint4 input_index1, input_index2, temp_index;
    uint dir, global_start, global_offset;
    uint temp_stage, temp_high_stage;
    uint4 comp;
    
    temp_high_stage = *high_stage;
    temp_stage = *stage;
 
    /* Determine location of data in global memory */
    dir = get_group_id(0)/temp_high_stage & 1;                                /*0/1/4/5/8/9/...升序, 2/3/6/7/10/11/...降序*/
    global_start = (get_group_id(0) + (get_group_id(0)/temp_stage)*temp_stage) * get_local_size(0) + get_local_id(0);
    global_offset = temp_stage * get_local_size(0);
    /*
    第0组第0个工作项，global_start=0*local_size+0
    第0组第1个工作项，global_start=0*local_size+1
    第0组第2个工作项，global_start=0*local_size+2
    ...
    第1组第0个工作项，global_start=1*local_size+0
    第1组第1个工作项，global_start=1*local_size+1
    第1组第2个工作项，global_start=1*local_size+2
    ...
    第2组第0个工作项，global_start=4*local_size+0
    第2组第1个工作项，global_start=4*local_size+1
    ...
    第3组第0个工作项，global_start=5*local_size+0
    第3组第1个工作项，global_start=5*local_size+1
    ...
    第4组第0个工作项，global_start=8*local_size+0
    第4组第1个工作项，global_start=8*local_size+1
    ...
    
    global_offset统一为2*local_size
    */

    /* Perform swap */
    input1 = g_data[global_start];
    input2 = g_data[global_start + global_offset];
    input_index1 = index_data[global_start];
    input_index2 = index_data[global_start + global_offset];
    SWAP_VECTORS(input1, input2, input_index1, input_index2, dir);
    g_data[global_start] = input1;
    g_data[global_start + global_offset] = input2;
    index_data[global_start] = input_index1;
    index_data[global_start + global_offset] = input_index2;
}


/* Perform lowest stage of the bitonic sort */
/*
在执行了阶段n以后，开始执行阶段0的操作

*/
__kernel void bsort_stage_0(__global float4 *g_data, __local float4 *l_data, __global uint4 *index_data, __local uint4 *l_index_data, __global uint *high_stage)
{

    float4 input1, input2, temp;
    uint4 input_index1, input_index2, temp_index;
    uint id, dir, global_start, stride;
    uint temp_high_stage;
    uint4 comp;
    
    temp_high_stage = *high_stage;

    /* Determine data location in global memory */
    id = get_local_id(0);
    dir = get_group_id(0)/temp_high_stage & 1;
    global_start = get_group_id(0) * get_local_size(0) * 2 + id;
 
    /* Perform initial swap */
    input1 = g_data[global_start];
    input2 = g_data[global_start + get_local_size(0)];
    input_index1 = index_data[global_start];
    input_index2 = index_data[global_start + get_local_size(0)];
    SWAP_VECTORS(input1, input2, input_index1, input_index2, dir);
    l_data[id] = input1;
    l_data[id + get_local_size(0)] = input2;
    l_index_data[id] = input_index1;
    l_index_data[id + get_local_size(0)] = input_index2;

    /* Perform bitonic merge */
    for(stride = get_local_size(0)/2; stride > 1; stride >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        id = get_local_id(0) + (get_local_id(0)/stride)*stride;
        SWAP_VECTORS(l_data[id], l_data[id + stride], l_index_data[id], l_index_data[id + stride], dir);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Perform final sort */
    id = get_local_id(0) * 2;
    input1 = l_data[id];
    input2 = l_data[id+1];
    input_index1 = l_index_data[id];
    input_index2 = l_index_data[id+1];
    SORT_TWO_VECTOR(input1, input2, input_index1, input_index2, dir);
 
    /* Store output in global memory */
    g_data[global_start + get_local_id(0)] = input1;
    g_data[global_start + get_local_id(0) + 1] = input2;
    index_data[global_start + get_local_id(0)] = input_index1;
    index_data[global_start + get_local_id(0) + 1] = input_index2;
}

/* Sort the bitonic set */
/*
经过前面的多次多阶段的分裂和排序，现在所有数据已经变为一个大的双调函数，现在要进行最后的分裂和排序
*/
__kernel void bsort_merge(__global float4 *g_data, __global uint4 *index_data, __global uint *stage, __global uint *dir)
{

    float4 input1, input2, temp;
    uint4 input_index1, input_index2, temp_index;
    uint global_start, global_offset;
    uint temp_stage,temp_dir;
    uint4 comp;
    
    temp_stage = *stage;
    temp_dir = *dir;

    /* Determine location of data in global memory */
    global_start = (get_group_id(0) + (get_group_id(0)/temp_stage)*temp_stage) * get_local_size(0) + get_local_id(0);
    global_offset = temp_stage * get_local_size(0);
 
    /* Perform swap */
    input1 = g_data[global_start];
    input2 = g_data[global_start + global_offset];
    input_index1 = index_data[global_start];
    input_index2 = index_data[global_start + global_offset];
    SWAP_VECTORS(input1, input2, input_index1, input_index2, temp_dir);
    g_data[global_start] = input1;
    g_data[global_start + global_offset] = input2;
    index_data[global_start] = input_index1;
    index_data[global_start + global_offset] = input_index2;
}

/* Perform final step of the bitonic merge */
/*
执行最后的混合(排序)
*/
__kernel void bsort_merge_last(__global float4 *g_data, __local float4 *l_data, __global uint4 *index_data, __local uint4 *l_index_data, __global uint *dir)
{

    uint id, global_start, stride;
    float4 input1, input2, temp;
    uint4 input_index1, input_index2, temp_index;
    uint temp_dir;
    uint4 comp;
    
    temp_dir = *dir;

    /* Determine location of data in global memory */
    id = get_local_id(0);
    global_start = get_group_id(0) * get_local_size(0) * 2 + id;

    /* Perform initial swap */
    input1 = g_data[global_start];
    input2 = g_data[global_start + get_local_size(0)];
    input_index1 = index_data[global_start];
    input_index2 = index_data[global_start + get_local_size(0)];
    SWAP_VECTORS(input1, input2, input_index1, input_index2, temp_dir);
    l_data[id] = input1;
    l_data[id + get_local_size(0)] = input2;
    l_index_data[id] = input_index1;
    l_index_data[id + get_local_size(0)] = input_index2;

    /* Perform bitonic merge */
    for(stride = get_local_size(0)/2; stride > 1; stride >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        id = get_local_id(0) + (get_local_id(0)/stride)*stride;
        SWAP_VECTORS(l_data[id], l_data[id + stride], l_index_data[id], l_index_data[id + stride], temp_dir);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Perform final sort */
    id = get_local_id(0) * 2;
    input1 = l_data[id];
    input2 = l_data[id+1];
    input_index1 = l_index_data[id];
    input_index2 = l_index_data[id+1];
    SORT_TWO_VECTOR(input1, input2, input_index1, input_index2, temp_dir);

    /* Store the result to global memory */
    g_data[global_start + get_local_id(0)] = input1;
    g_data[global_start + get_local_id(0) + 1] = input2;
    index_data[global_start + get_local_id(0)] = input_index1;
    index_data[global_start + get_local_id(0) + 1] = input_index2;
}





/*
ERROR!!!!!!!!!!!!!!!!!!!!!!


得到矩阵A中每一行数据（作为向量）到某个向量B的距离，A的列数和B的元素个数一样，结果D是一个一维数组，长度与A的行数一样

A dist B = D

|a,b,c,d|                   | ( (a-w)^2+(b-x)^2+(c-y)^2+(d-z)^2 )^0.5 |
|e,f,g,h|                   | ( (e-w)^2+(f-x)^2+(g-y)^2+(h-z)^2 )^0.5 |
|i,j,k,l| dist  |w,x,y,z| = | ( (i-w)^2+(j-x)^2+(k-y)^2+(l-z)^2 )^0.5 |
|m,n,o,p|                   | ( (m-w)^2+(n-x)^2+(o-y)^2+(p-z)^2 )^0.5 |
其中A中的元素都是float4的
数据划分上，规定数据是2维，每个工作组先处理一个元素（a,b,c...），将元素减去对应的B中的元素。得到
| (a-w) (b-x) (c-y) (d-z) |
| (e-w) (f-x) (g-y) (h-z) |
| (i-w) (j-x) (k-y) (l-z) |
| (m-w) (n-x) (o-y) (p-z) |
然后再将新矩阵中的每个位置上平方，得到
| (a-w)^2 (b-x)^2 (c-y)^2 (d-z)^2 |
| (e-w)^2 (f-x)^2 (g-y)^2 (h-z)^2 |
| (i-w)^2 (j-x)^2 (k-y)^2 (l-z)^2 |
| (m-w)^2 (n-x)^2 (o-y)^2 (p-z)^2 |
然后每个gid(0)==0的工作项负责将各个行的数据相加开方
| ((a-w)^2 (b-x)^2 (c-y)^2 (d-z)^2)^0.5 |
| ((e-w)^2 (f-x)^2 (g-y)^2 (h-z)^2)^0.5 |
| ((i-w)^2 (j-x)^2 (k-y)^2 (l-z)^2)^0.5 |
| ((m-w)^2 (n-x)^2 (o-y)^2 (p-z)^2)^0.5 |
计算结束

ERROR!!!!!!!!!!!!!!!!!!!!!!
*/


