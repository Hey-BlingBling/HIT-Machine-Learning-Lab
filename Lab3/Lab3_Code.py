import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import random

def gen_2d_data(number_list,mean_list,cov_list):
    '''
    生成k类二维数据

    Parameters
    ----------
    number_list : TYPE
        k类数据的数量，以列表形式给出
    mean_list : TYPE
        k类数据的均值，以列表形式给出
    cov_list : TYPE
        k类数据的协方差矩阵，以列表形式给出

    Returns
    -------
    TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.

    '''
    assert len(number_list)==len(mean_list)
    assert len(mean_list)==len(cov_list)
    k=len(number_list)
    origin_data=[]
    data=[]
    for i in range(k):
        data_i=np.random.multivariate_normal([mean_list[i][0],mean_list[i][1]],cov_list[i],number_list[i],'raise').tolist()
        origin_data.append(data_i)
        data=data+data_i
    return np.array(origin_data),np.array(data)

def plot_2d(data,title):
    '''
    画出数据对于的图像

    Parameters
    ----------
    data : TYPE
        数据
    title : TYPE
        图像标题

    Returns
    -------
    None.

    '''
    color_dic={0:'red',1:'blue',2:'green',3:'purple',4:'yellow'}
    plt.xlabel('x')
    plt.ylabel('y')
    for i in range(len(data)):
        for j in range(len(data[i])):
            plt.scatter(data[i][j][0],data[i][j][1],
                        color=color_dic[i],marker='.')
    plt.title(title)
    plt.show()

def k_means(data,k,max_time):
    '''
    K-Means算法聚类

    Parameters
    ----------
    data : TYPE
        需要进行聚类的数据
    k : TYPE
        聚类的数量
    max_time : TYPE
        迭代进行的最大次数

    Returns
    -------
    ans : TYPE
        分类后的结果，是k个列表组合成的列表，每个子列表中存放一簇数据
    times : TYPE
        迭代进行的次数
    test_type : TYPE
        对数据分类后的分组结果

    '''
    mu=[]#存放k类的均值
    init_mu_index=random.sample(range(1,len(data)),k)#随机选取初始点
    for i in init_mu_index:
        mu.append(data[i])
    times=0
    while(times<max_time):
        times=times+1
        ans=[]
        for i in range(k):#每次更新均值后要重新初始化数据矩阵
            ans_i=[]
            ans.append(ans_i)
        is_change=[0]*k
        for x_j in data:#计算每个样本点到每个均值的距离，并将其加入到最近的簇内
            distance=[]
            for mu_i in mu:
                distance.append(np.linalg.norm(x_j-mu_i))
            lam_j=distance.index(min(distance))
            ans[lam_j].append(x_j)
        index=0
        for ans_i in ans:#重新计算簇内均值
            sum_i=[0,0]
            for j in range(len(ans_i)):
                sum_i=sum_i+ans_i[j]
            average_i=np.array(sum_i)/len(ans_i)
            if any(average_i!=mu[index]):
                mu[index]=average_i
                is_change[index]=1
            index+=1
        if not any(is_change):#每个簇的均值不再发生改变时即可结束循环
            break
    return ans,times

def gaussian_mixture_model(data,k,max_time):
    '''
    高斯混合聚类

    Parameters
    ----------
    data : TYPE
        数据
    k : TYPE
        聚类个数
    max_time : TYPE
        最大迭代次数

    Returns
    -------
    ans : TYPE
        分类后的数据
    times : TYPE
        迭代次数
    test_type : TYPE
        分类结果

    '''
    alpha=[1/k]*k#混合成分，均初始化为1/k
    mu=[]#存放k类的均值
    init_mu_index=random.sample(range(1,len(data)),k)#随机选取初始点
    for i in init_mu_index:
        mu.append(data[i])
    mu=np.array(mu)
    #sigma=[[[1,0],[0,1]]]*k#初始化协方差矩阵
    sigma=[np.identity(data.shape[1])]*k
    gamma_ji=np.zeros((len(data),k))#gamma_ji[j][i]表示x_j由混合成分alpha_i生成的概率
    times=0
    while(times<max_time):
        for j in range(len(data)):
            for i in range(k):
                top=alpha[i]*multivariate_normal.pdf(data[j],mu[i],sigma[i])
                bottom=0
                for l in range(k):
                    bottom+=alpha[l]*multivariate_normal.pdf(data[j],mu[l],sigma[l])
                gamma_ji[j][i]=top/bottom
        for i in range(k):
            mu_top=np.zeros(len(data[0]))
            sigma_top=np.zeros((len(data[0]),len(data[0])))
            bottom=0
            for j in range(len(data)):
                mu_top+=gamma_ji[j][i]*data[j]
                bottom+=gamma_ji[j][i]
            mu[i]=mu_top/bottom
            for j in range(len(data)):
                temp=np.array(data[j]-mu[i]).reshape((len(data[0]),1))
                sigma_top+=gamma_ji[j][i]*np.dot(temp,temp.T)
            sigma[i]=sigma_top/bottom
            alpha[i]=bottom/len(data)
        times+=1
    ans=[]
    for i in range(k):
        ans_i=[]
        ans.append(ans_i)
    test_type=[]
    for j in range(len(data)):
        index=np.argmax(gamma_ji[j])
        test_type.append(index)
        ans[index].append(data[j])
    return ans,times,test_type

def UCI_test(path):
    '''
    UCI测试

    Parameters
    ----------
    path : TYPE
        测试数据的路径

    Returns
    -------
    None.

    '''
    f=open(path,'r')
    lines=f.readlines()
    temp=[]
    real_type=[]
    for line in lines:
        line=line.split(',')
        l=[]
        l.append(float(line[0]))
        l.append(float(line[1]))
        l.append(float(line[2]))
        l.append(float(line[3]))
        real_type.append(float(line[4]))
        temp.append(l)
    temp=np.array(temp)
    ans_GMM,times_GMM,test_type=gaussian_mixture_model(temp, 2, 50)
    correct_num=0
    total_num=len(temp)
    for i in range(total_num):
        if real_type[i]==test_type[i]:
            correct_num+=1
    print('Accuracy:',correct_num/total_num if correct_num/total_num>0.5 else 1-correct_num/total_num)

number_list=[100,110,120]
mean_list=[[0,1],[0,6],[4,4]]
cov_list=[[[0.5,0],[0,0.5]],[[0.5,0],[0,0.5]],[[0.5,0],[0,0.5]]]
origin_data,data=gen_2d_data(number_list, mean_list, cov_list)
plot_2d(origin_data,'Original Data')
ans,times=k_means(data, 3, 10)
plot_2d(ans,'K - Means, times = '+str(times))
ans_GMM,times_GMM,test_type=gaussian_mixture_model(data, 3, 10)
plot_2d(ans_GMM,'GMM, times = '+str(times_GMM))
UCI_test('data.txt')