import numpy as np
import matplotlib.pyplot as plt

#拟合的多项式的系数
global polyDegree
polyDegree=10

def genSample(mu,sigma,sampleNum):
    '''
    生成测试集和训练集的数据

    Parameters
    ----------
    mu : TYPE
        数据的均值
    sigma : TYPE
        数据的方差
    sampleNum : TYPE
        生成的数据的数量

    Returns
    -------
    x : TYPE
        生成的样本的x坐标
    y : TYPE
        生产的样本的y坐标

    '''
    x=np.arange(0,1,1/sampleNum);
    gaussNoise=np.random.normal(mu,sigma,sampleNum)
    y=np.sin(x*2*np.pi)+gaussNoise
    return x,y

def genMatrixX(trainX):
    '''
    将生成的训练集数据坐标转换为对应的矩阵形式

    Parameters
    ----------
    trainX : TYPE
        训练集数据

    Returns
    -------
    X : TYPE
        转换后的矩阵

    '''
    X=np.zeros((len(trainX),polyDegree+1))
    for i in range(len(trainX)):
        row=np.ones(polyDegree+1)*trainX[i]
        rowP=np.arange(0,polyDegree+1)
        row=np.power(row,rowP)
        X[i]=row
    return X

def loss(trainX,trainY,w):
    '''
    计算在当前数据集下以及当前拟合的多项式下的误差

    Parameters
    ----------
    trainX : TYPE
        数据集的横坐标矩阵
    trainY : TYPE
        数据集的纵坐标
    w : TYPE
        多项式的系数

    Returns
    -------
    loss : TYPE
        计算出的误差值

    '''
    X=genMatrixX(trainX)
    Y=trainY.reshape(len(trainX),1)
    tempM=X.dot(w)-Y
    #print(tempM)
    loss=1/2*np.dot(tempM.T,tempM)
    #print(loss)
    return loss

def myPolyfit(trainX,trainY):
    '''
    不带惩罚项的最小二乘拟合

    Parameters
    ----------
    trainX : TYPE
        数据集的横坐标矩阵
    trainY : TYPE
        数据集的纵坐标

    Returns
    -------
    w : TYPE
        拟合出的多项式系数
    poly : TYPE
        系数对应的多项式

    '''
    #print(polyDegree)
    X=genMatrixX(trainX)
    Y=trainY.reshape(len(trainX),1)
    w=np.linalg.inv(np.dot(X.T,X)).dot(X.T).dot(Y)
    #print(w)
    poly=np.poly1d(w[::-1].reshape(polyDegree+1))
    #print(poly)
    return w,poly

def punishPolyfit(trainX,trainY,lam):
    '''
    加入惩罚项后的最小二乘拟合

    Parameters
    ----------
    trainX : TYPE
        数据集的横坐标矩阵
    trainY : TYPE
        数据集的纵坐标
    lam : TYPE
        惩罚项系数

    Returns
    -------
    w : TYPE
        拟合出的多项式系数
    poly : TYPE
        系数对应的多项式

    '''
    X=genMatrixX(trainX)
    Y=trainY.reshape(len(trainX),1)
    w=np.linalg.inv(np.dot(X.T,X)+lam*np.eye(X.shape[1])).dot(X.T).dot(Y)
    poly=np.poly1d(w[::-1].reshape(polyDegree+1))
    return w,poly

def gradientDescent(trainX,trainY,lam,maxTime,stepSize,eps):
    '''
    使用梯度下降法进行优化拟合

    Parameters
    ----------
    trainX : TYPE
        数据集的横坐标矩阵
    trainY : TYPE
        数据集的纵坐标
    lam : TYPE
        惩罚项系数
    maxTime : TYPE
        最大的迭代次数
    stepSize : TYPE
        迭代的步长
    eps : TYPE
        最小迭代误差

    Returns
    -------
    w : TYPE
        拟合出的多项式系数
    poly : TYPE
        系数对应的多项式

    '''
    w=0.1*np.ones((polyDegree+1,1))
    X=genMatrixX(trainX)
    Y=trainY.reshape((len(trainX),1))
    timeL=np.zeros(maxTime)

    for i in range(maxTime):
        lossBeforeCal=abs(loss(trainX,trainY,w))
        partialDeriv=X.T.dot(X).dot(w)-X.T.dot(Y)+lam*w
        w=w-stepSize*partialDeriv
        lossAfterCal=abs(loss(trainX,trainY,w))
        timeL[i]=i
        if(lossAfterCal>lossBeforeCal):
            stepSize=stepSize/2
        if(abs(lossAfterCal-lossBeforeCal)<eps):
            break
    poly=np.poly1d(w[::-1].reshape(polyDegree+1))
    return w,poly

def conjugateGradient(trainX,trainY,lam,eps):
    '''
    使用共轭梯度法进行拟合优化

    Parameters
    ----------
    trainX : TYPE
        数据集的横坐标矩阵
    trainY : TYPE
        数据集的纵坐标
    lam : TYPE
        惩罚项系数
    eps : TYPE
        最小迭代误差

    Returns
    -------
    w : TYPE
        拟合出的多项式系数
    poly : TYPE
        系数对应的多项式

    '''
    X=genMatrixX(trainX)
    Y=trainY.reshape((len(trainX),1))
    Q=np.dot(X.T,X)+lam*np.eye(X.shape[1])
    w=np.zeros((polyDegree+1,1))
    gradient=np.dot(X.T,X).dot(w)-np.dot(X.T,Y)+lam*w
    r=-gradient
    p=r
    for i in range(polyDegree+1):
        alpha=(r.T.dot(r))/(p.T.dot(Q).dot(p))
        rBeforeCal=r
        w=w+alpha*p
        r=r-(alpha*Q).dot(p)
        beta=(r.T.dot(r))/(rBeforeCal.T.dot(rBeforeCal))
        p=r+beta*p
    poly=np.poly1d(w[::-1].reshape(polyDegree+1))
    return w,poly

#图像表示
def plot2(trainX,trainY,polyAns,title):
    '''
    对拟合的数据绘制对应的图像

    Parameters
    ----------
    trainX : TYPE
        数据集的横坐标矩阵
    trainY : TYPE
        数据集的纵坐标
    polyAns : TYPE
        拟合出的多项式
    title : TYPE
        绘制的图像的标题

    Returns
    -------
    None.

    '''
    #样本点
    plt.plot(trainX,trainY,'s',label="Sample Data")

    #标准曲线绘制
    normX=np.arange(0,1,1/100)
    normY=np.sin(normX*2*np.pi)
    plt.plot(normX,normY,'b',label='Norm Data')

    #拟合曲线绘制
    fitY=polyAns(normX)
    plt.plot(normX,fitY,'r',label='Fit Data')

    plt.axis([-0.01,1.01,-1.5,1.5])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right')
    plt.title(title)
    plt.show()

def lossAndDegree():
    '''
    随着阶数的增高，loss的变化曲线

    Returns
    -------
    None.

    '''
    global polyDegree
    trainSize=15
    testSize=60
    trainX,trainY=genSample(0,0.1,trainSize)
    testX,testY=genSample(0,0.1,testSize)
    x=[]
    y1=[]
    y2=[]
    max=10
    for i in range(1,max):
        x.append(i)
        polyDegree=i
        w,poly=myPolyfit(trainX, trainY)
        #plot2(trainX, trainY, poly,'fig')
        #print(loss(trainX,trainY,w))
        y1.append(float(loss(trainX,trainY,w)))
        y2.append(float(loss(testX,testY,w)))
        #print(x)
        #print(y)
    title='train set size = ' + str(trainSize) + ', test set size = ' + str(testSize)
    plt.axis([0,9,0,5])
    plt.plot(x,y1,'b',label='train data')
    plt.plot(x,y2,'r',label='test data')
    plt.xlabel('degree')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.title(title)
    plt.show()

def lossAndSize():
    '''
    改变训练集的大小对模型的影响

    Returns
    -------
    None.

    '''
    global polyDegree
    polyDegree=5
    trainList=[]
    lossList=[]
    testX,testY=genSample(0,0.1,100)
    for i in range(10,1000,100):
        trainList.append(i)
        trainX,trainY=genSample(0,0.1,i)
        w,poly=myPolyfit(trainX, trainY)
        lossList.append(float(loss(testX,testY,w)))
    plt.plot(trainList,lossList,'r')
    title='degree = ' + str(polyDegree)
    plt.xlabel('train set size')
    plt.ylabel('loss')
    plt.title(title)
    plt.show()

def lossAndLam():
    '''
    改变惩罚项系数对于模型的影响

    Returns
    -------
    None.

    '''
    global polyDegree
    polyDegree=5
    sampleNum=10
    lamList=[]
    lossList=[]
    trainX,trainY=genSample(0,0.1,sampleNum)
    testX,testY=genSample(0,0.1,1000)
    for i in range(-20,0,1):
        lam=np.power(np.e,i)
        #print(lam)
        w,poly=punishPolyfit(trainX, trainY, lam)
        lamList.append(i)
        lossList.append(float(loss(testX,testY,w)))
    plt.plot(lamList,lossList,'r')
    title='degree = ' + str(polyDegree)
    plt.xlabel('ln($\lambda $)')
    plt.ylabel('loss')
    plt.title(title)
    plt.show()

def main():
    global polyDegree

    #生成样本点
    trainX,trainY=genSample(0,0.15,10)

    #使用最小二乘拟合的结果
    for i in range(1,10):
        polyDegree=i
        w,myPloyAns=myPolyfit(trainX, trainY)
        #print(myPloyAns)
        plot2(trainX, trainY, myPloyAns, 'degree = ' + str(polyDegree))

    #加入惩罚项的拟合结果
    lam1=0.0005
    for i in range(1,10):
        polyDegree=i
        w,punishPloyAns=punishPolyfit(trainX, trainY, lam1)
        plot2(trainX, trainY, punishPloyAns, 'degree = ' + str(polyDegree) + ', $\lambda $ = ' +str(lam1))

    #使用梯度下降法的拟合结果
    lam2=0.0001
    maxTime=100000
    stepSize=0.5
    eps1=1e-6
    for i in range(1,10):
        polyDegree=i
        w,gradientDescentAns=gradientDescent(trainX, trainY, lam2, maxTime, stepSize, eps1)
        plot2(trainX, trainY, gradientDescentAns, 'degree = ' + str(polyDegree)
              + ', $\lambda $ = ' + str(lam2)
              + ', $\eta $ = ' + str(stepSize)
              + ', $\epsilon $ = ' + str(eps1)
              + ', maxTime = ' + str(maxTime)
              )

    #使用共轭梯度法的拟合结果
    lam3=0.0005
    eps2=1e-5
    for i in range(1,10):
        polyDegree=i
        w,conjugateGradientAns=conjugateGradient(trainX,trainY,lam3,eps2)
        plot2(trainX, trainY, conjugateGradientAns, 'degree = ' + str(polyDegree)
              + ', $\lambda $ = ' + str(lam3)
              + ', $\epsilon $ = ' + str(eps2)
              )

    #loss与多项式系数的关系曲线
    lossAndDegree()
    #loss与惩罚项系数的关系曲线
    lossAndLam()
    #loss和训练集大小的关系曲线
    lossAndSize()


if __name__=='__main__':
    main()