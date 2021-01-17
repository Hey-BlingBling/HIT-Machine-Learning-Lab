import numpy as np
import matplotlib.pyplot as plt

global up,down,left,right
global SIZE
SIZE=500

def genData(mu0,sig0,size0,mu1,sig1,size1):
    '''
    Parameters
    ----------
    mu0 : TYPE
        class0各维度的均值，应该是一个长度为n的行向量，n是特征数目。
    sig0 : TYPE
        class0各维度的方差，应该是一个长度为n的行向量，n是特征数目。
    size0 : TYPE
        class0的样本数量。
    mu1 : TYPE
        class1各维度的均值，应该是一个长度为n的行向量，n是特征数目。
    sig1 : TYPE
        class1各维度的方差，应该是一个长度为n的行向量，n是特征数目。
    size1 : TYPE
         class1的样本数量。
         
    Returns
    -------
    TYPE
        生成的满足朴素贝叶斯假设的样本的各维度数据，是一个(size0+size1)行n列的矩阵。
    TYPE
        生成的满足朴素贝叶斯假设的样本所对应的类别，是一个m行的列向量。
    '''
    global up,down,left,right
    
    x0=np.empty((size0,mu0.size))
    x1=np.empty((size1,mu1.size))
    class0=np.zeros((size0,1)).reshape((size0,1))
    class1=np.ones((size1,1)).reshape((size1,1))
    for j in range(mu0.size):
        for i in range(size0):
            x0[i][j]=np.random.normal(mu0[j],sig0[j])
    for j in range(mu1.size):
        for i in range(size1):    
            x1[i][j]=np.random.normal(mu1[j],sig1[j])
    x0=np.column_stack((x0,class0))
    x1=np.column_stack((x1,class1))
    x=np.row_stack((x0,x1))
    np.random.shuffle(x)
    
    left=x[:,0].min()
    right=x[:,0].max()
    up=x[:,1].max()
    down=x[:,1].min()
    
    return x[:,:-1].reshape((size0+size1),mu0.size),x[:,-1].reshape((size0+size1),1)

def gen2dRelatedData(mu0,size0,mu1,size1,cov00,cov01,cov10,cov11):
    '''
    Parameters
    ----------
    mu0 : TYPE
        class0各维度的均值，应该是一个长度为n的行向量，n是特征数目。
    size0 : TYPE
        class0的样本数量。
    m1 : TYPE
        class1各维度的均值，应该是一个长度为n的行向量，n是特征数目。
    size1 : TYPE
        class1的样本数量。
    cov00 : TYPE
        特征0的方差。
    cov01 : TYPE
        特征0和特征1的协方差。
    cov10 : TYPE
        特征1和特征0的协方差。
    cov11 : TYPE
        特征1的方差。
        
    Returns
    -------
    TYPE
        生成的不满足朴素贝叶斯假设的样本的各维度数据，是一个(size0+size1)行n列的矩阵。
    TYPE
        生成的不满足朴素贝叶斯假设的样本所对应的类别，是一个m行的列向量。
    '''
    x0=np.empty((size0,mu0.size))
    x1=np.empty((size1,mu1.size))
    class0=np.zeros((size0,1)).reshape((size0,1))
    class1=np.ones((size1,1)).reshape((size1,1))
    x0=np.random.multivariate_normal(mu0,np.array([[cov00,cov01],[cov10,cov11]]),size=size0)
    x1=np.random.multivariate_normal(mu1,np.array([[cov00,cov01],[cov10,cov11]]),size=size1)
    x0=np.column_stack((x0,class0))
    x1=np.column_stack((x1,class1))
    x=np.row_stack((x0,x1))
    np.random.shuffle(x)
    return x[:,:-1].reshape((size0+size1),mu0.size),x[:,-1].reshape((size0+size1),1)

def loss(x,y,w,lam):
    '''
    Parameters
    ----------
    x : TYPE
        增广后的m行n+1列的数据矩阵，其中m为数据个数，n为数据特征数。
    y : TYPE
        x中所有数据对应的类别，是一个m行的列向量。
    w : TYPE
        增广后的系数向量，是一个n+1行的列向量。
    lam : TYPE
        惩罚项系数，为0时表示不需要惩罚项
        
    Returns
    -------
    loss : TYPE
        该数据集在指定模型下的损失值。
    '''
    y=1-y
    lossSum=0
    for i in range(y.shape[0]):
        lossSum+=(y[i]*np.dot(x[i],w)-np.log(1+np.e**(np.dot(x[i],w))))
    lossSum=-lossSum/y.shape[0]
    lossSum+=lam*float(np.dot(w.T,w))/(2*y.shape[0])
    return lossSum

def gradientDescent(trainData,trainCla,lam,alpha=0.05,epochs=10000,eps=1e-5):
    '''
    Parameters
    ----------
    trainData : TYPE
        m行n列的数据矩阵，其中m为数据个数，n为数据特征数。
    trainCla : TYPE
        trainData中所有数据对应的类别，是一个m行的列向量。
    lam : TYPE
        惩罚项系数。
    alpha : TYPE, optional
        DESCRIPTION. The default is 0.05.
        学习率。
    epochs : TYPE, optional
        DESCRIPTION. The default is 10000.
        最大迭代次数。
    eps : TYPE, optional
        DESCRIPTION. The default is 1e-5.
        最小误差。

    Returns
    -------
    w : TYPE
        优化后求出的系数向量。
    '''
    temp=np.ones((trainData.shape[0],1))
    trainData=np.column_stack((temp,trainData))
    
    w=0.1*np.ones((trainData.shape[1],1)).reshape((trainData.shape[1],1))
    
    for i in range(epochs):
        lossBeforeCal=abs(loss(trainData,trainCla,w,lam))
        partialDeriv=0
        for j in range(trainCla.shape[0]):
            partialDeriv+=trainData[j]*((1-trainCla[j])-(np.e**(np.dot(trainData[j],w)))/(1+np.e**(np.dot(trainData[j],w))))
        partialDeriv=-partialDeriv/trainCla.shape[0]
        partialDeriv=partialDeriv.reshape((trainData.shape[1],1))
        partialDeriv+=lam*w/trainCla.shape[0]
        w=w-alpha*partialDeriv
        lossAfterCal=abs(loss(trainData,trainCla,w,lam))
        if(lossAfterCal>lossBeforeCal):
            alpha/=2
        if(abs(lossAfterCal-lossBeforeCal)<eps):
            break
    return w

def calAccuracy(testData,testCla,w):
    '''
    Parameters
    ----------
    testData : TYPE
        测试数据。
    testCla : TYPE
        测试数据所属类。
    w : TYPE
        优化后求出的系数向量。

    Returns
    -------
    int
        在指定系数向量下计算出的测试数据的分类准确率。
    '''
    temp=np.ones((testData.shape[0],1))
    testData=np.column_stack((temp,testData))
    
    pridictCla=1/(1+np.e**np.dot(testData,w))
    rightNum=0
    for i in range(testCla.size):
        if (pridictCla[i]>=0.5) and (testCla[i]==1):
            rightNum+=1
        if (pridictCla[i]<0.5) and (testCla[i]==0):
            rightNum+=1
    return round(rightNum/testCla.size,6),rightNum,testCla.size

def plot2(x,y,w,title):
    '''
    Parameters
    ----------
    x : TYPE
        需要绘制的数据信息。
    y : TYPE
        需要绘制的数据的类别。
    w : TYPE
        优化后求出的系数向量。
    title : TYPE
        绘制的图像标题。

    Returns
    -------
    None.
    '''
    global up,down,left,right
    
    setSize=x.shape[0]
    accuracy,rightNum,totalNum=calAccuracy(x,y,w)
    title+=' set size = '+str(setSize)+', Accuracy = '+str(accuracy)
    
    feature1=x[:,0]
    feature2=x[:,1]
    plt.xlabel('Feature1')
    plt.ylabel('Feature2')
    for i in range(x.shape[0]):
        if(y[i]==0):
            plt.scatter(feature1[i],feature2[i],color='b',marker='.')
        if(y[i]==1):
            plt.scatter(feature1[i],feature2[i],color='r',marker='.')
    x=np.arange(0,10,1/100)
    b=-w[0]/w[2]
    k=-w[1]/w[2]
    plt.plot(x,k*x+b,color='g')
    plt.title(title)
    plt.axis([left-0.5,right+0.5,down-0.5,up+0.5])
    plt.show()
    
def UCITest(path):
    '''
    Parameters
    ----------
    path : TYPE
        读取的UCI文件的路径。

    Returns
    -------
    None.

    '''
    f=open(path,'r')
    lines=f.readlines()
    temp=[]
    for line in lines:
        line=line.split(',')
        f=[]
        f.append(float(line[0]))
        f.append(float(line[1]))
        f.append(float(line[2]))
        f.append(float(line[3]))
        f.append(float(line[4]))
        temp.append(f)
        
    temp=np.array(temp)
    
    trainSize=int(len(lines)*0.7)
    train=temp[0:trainSize,:]
    test=temp[trainSize:len(lines),:]
    #train=np.array(train)
    #test=np.array(test)
    trainData=train[:,0:4]
    trainCla=train[:,-1]
    testData=test[:,0:4]
    testCla=test[:,-1]
    
    w=gradientDescent(trainData,trainCla,0.002)
    
    accTrain,rightNumTrain,totalNumTrain=calAccuracy(trainData, trainCla, w)
    accTest,rightNumTest,totalNumTest=calAccuracy(testData, testCla, w)
    
    print('在训练集上的正确样本数为'+str(rightNumTrain)+',样本总数为'+str(totalNumTrain)+',准确率为'+str(accTrain))
    print('在测试集上的正确样本数为'+str(rightNumTest)+',样本总数为'+str(totalNumTest)+',准确率为'+str(accTest))

def unrelatedDataTest():
    '''
    对于满足朴素贝叶斯假设的数据的测试。
    
    Returns
    -------
    None.

    '''
    size0=SIZE
    size1=SIZE
    mu0=np.array([1,1])
    sig0=np.array([1,1])
    mu1=np.array([4,4])
    sig1=np.array([1,1])
    allData,allCla=genData(mu0,sig0,size0,mu1,sig1,size1)
    trainSize=int((size0+size1)*0.7)
    trainData=allData[0:trainSize,:]
    trainCla=allCla[0:trainSize,:]
    testData=allData[trainSize:size0+size1,:]
    testCla=allCla[trainSize:size0+size1,:]
    w=gradientDescent(trainData,trainCla,0.002)
    plot2(trainData,trainCla,w,'Unrelated Data Result, Train')
    plot2(testData,testCla,w,'Unrelated Data Result, Test')
    
def relatedDataTest():
    '''
    对于不满足朴素贝叶斯假设的数据的测试。
    
    Returns
    -------
    None.

    '''
    size0=SIZE
    size1=SIZE
    mu0=np.array([1,1])
    mu1=np.array([4,4])
    allData,allCla=gen2dRelatedData(mu0,size0,mu1,size1,1,0.2,0.2,1)
    trainSize=int((size0+size1)*0.7)
    trainData=allData[0:trainSize,:]
    trainCla=allCla[0:trainSize,:]
    testData=allData[trainSize:size0+size1,:]
    testCla=allCla[trainSize:size0+size1,:]
    w=gradientDescent(trainData,trainCla,0.002)
    plot2(trainData,trainCla,w,'Related Data Result, Train')
    plot2(testData,testCla,w,'Related Data Result, Test')

#满足朴素贝叶斯假设的测试
unrelatedDataTest()
#不满足朴素贝叶斯假设的测试
relatedDataTest()
#UCI数据的测试
UCITest('data.txt')