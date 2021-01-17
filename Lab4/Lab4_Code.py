import numpy as np
import os
from matplotlib import pyplot as plt
from os import path
import cv2

reshape_size=(80,80)

def gen_3d_data(num):
    mean=[6,7,8]
    cov=[[0.3,0,0],[0,0.4,0],[0,0,0.5]]
    return np.array(np.random.multivariate_normal(mean,cov,num).tolist())

def pca(data,tar_dimension):
    data_num,dimension=data.shape
    print('Origin Dimension:',dimension)
    print('Target Dimension:',tar_dimension)
    mean=1/data_num*np.sum(data)
    trans_data=data-mean
    cov=trans_data.T.dot(trans_data)
    eig,vec=np.linalg.eig(cov)
    sort_ans=np.argsort(eig)
    vec=np.delete(vec,sort_ans[:dimension-tar_dimension],axis=1)

    recovery_data=(data-mean).dot(vec).dot(vec.T)+mean

    print('PSNR:',np.real(psnr(data,recovery_data)))

    return recovery_data

def plot_3d(origin_data,dimensional_data):
    fig=plt.figure()
    ax=fig.gca(projection='3d')
    ax.scatter(origin_data[:,0],origin_data[:,1],origin_data[:,2],facecolor='r',label='Origin Data')
    ax.scatter(dimensional_data[:,0],dimensional_data[:,1],dimensional_data[:,2],facecolor='b',label='Dimensional Data')
    plt.legend(prop={'size':8})
    plt.show()

def psnr(source, target):
    diff=source-target
    diff=diff**2
    rmse=np.sqrt(np.mean(diff))
    return 20*np.log10(255.0/rmse)

def read_face_data(file_path):
    global reshape_size

    face_data=[]
    file_names=os.listdir(file_path)
    plt.figure(figsize=reshape_size)
    i=0
    for f in file_names:
        plt.subplot(1,5,i+1)
        i+=1
        image_name=path.join(file_path,f)
        image=cv2.imread(image_name)
        image=cv2.resize(image,reshape_size)
        gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        plt.imshow(gray_image)
        length,width=gray_image.shape
        gray_image=gray_image.reshape(length*width)
        face_data.append(gray_image)
    plt.show()
    return np.array(face_data)

def face_data_pca(origin_face_data):
    global reshape_size

    d=100
    dimensional_face_data = pca(origin_face_data, d)
    plt.figure(figsize=reshape_size)
    for i in range(5):
        image=np.real(dimensional_face_data[i]).reshape(reshape_size)
        plt.subplot(1,5,i+1)
        plt.imshow(image)
    plt.show()
'''
origin_data=gen_3d_data(100)
dimensional_data=pca(origin_data,2)
plot_3d(origin_data, dimensional_data)
'''
face_data=read_face_data('Face_Data')
face_data_pca(face_data)