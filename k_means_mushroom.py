from numpy import *

def LoadData(filename):
    with open(filename,'r')as file:
        dataset=[line.strip().split(' ') for line in file.readlines()]
    dataset=[[int(column) if column.isdigit() else column for column in row ]for row in dataset]#将数据集中所有的字符数据转换为数字
    dataset = [row[1:-1] for row in dataset]#去除第一行的标签

    return array(dataset)#借助Numpy.array类型来实现数据集的操作简单化

def RandCentroid(dataset,k):#产生随机的k个质心
    length=shape(dataset)[1]
    centroid=mat(zeros((k,length)))
    for i in range(length):#对于dataset的每一个列，找出最小值和最大值，range代表最小值和最大值的差值，随机数可以设置为最小值加上差值*n,n小于1
        minnum=min(dataset[:,i])
        dis=float(max(dataset[:,i])-minnum)
        centroid[:,i]=minnum+dis*random.rand(k,1)
    return centroid

def k_means(DataSet,k):

    centroid = RandCentroid(DataSet,k)#得到随机的K个质心
    length = shape(DataSet)[0]#数据集的行数
    changed = True
    cluster = mat(zeros((length,2)))#一个数据集行数*2大小的矩阵，第一列用来保存当前行的向量所归属的质心的索引，第二列用来保存当前行的向量距离当前所归属的质心的距离

    while changed:
        changed=False
        for i in range(length):
            min_distance=Inf#设置一个最小距离和该最小距离对应的质心的索引
            index = -1
            for j in range(k):
                distance = sqrt(sum(power(centroid[j,:]-DataSet[i,:],2)))#计算两个变量的距离，即第j个质心和第i个数据的距离，复杂度为O（ij）
                if distance < min_distance:#如果又一个更短的距离，说明当前的数据不应该划分到这个质心上，应该改变质心
                    min_distance = distance
                    index = j

            if cluster[i,0]!=index:#如果新的质心和旧的质心不相等，则设置修改变量为True，即质心仍旧发生了改变
                changed = True
            cluster[i,:] = index,distance**2#修改新的质心索引和质心距离，用来给下次的计算进行比较
        for i in range(k):#用来更新新的质心的坐标
            p = DataSet[nonzero(cluster[:,0].A == i)[0]]
            centroid[i,:] = mean(p,axis = 0)
    return centroid,cluster

DataSet = LoadData(r"C:\Users\yang\Desktop\mushroom.dat")
centroid,cluster = k_means(DataSet,10)
print(centroid)
