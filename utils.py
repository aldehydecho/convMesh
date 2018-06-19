import scipy.io as sio
import numpy as np

def load_data(path):
    resultmax = 0.95
    resultmin = -0.95
    
    data = sio.loadmat(path)
    logr = data['FLOGRNEW']
    s = data['FS']
    
    pointnum=logr.shape[1]

    logrmin = logr.min()
    logrmin = logrmin - 1e-6
    logrmax = logr.max()
    logrmax = logrmax + 1e-6
    smin = s.min()
    smin = smin- 1e-6
    smax = s.max()
    smax = smax + 1e-6
    
    rnew = (resultmax-resultmin)*(logr-logrmin)/(logrmax - logrmin) + resultmin
    snew = (resultmax-resultmin)*(s - smin)/(smax-smin) + resultmin
    
    feature = np.concatenate((rnew,snew),axis = 2)
    
    f = np.zeros_like(feature).astype('float32')
    f = feature
    
    return f, logrmin, logrmax, smin, smax, pointnum

def load_neighbour(path, name, pointnum):
    data = sio.loadmat(path)
    data = data[name]
    
    maxdegree = data.shape[1]
    
    neighbour = np.zeros((pointnum, maxdegree)).astype('float32')
    neighbour = data

    degree = np.zeros((neighbour.shape[0], 1)).astype('float32')
    
    for i in range(neighbour.shape[0]):
        degree[i] = np.count_nonzero(neighbour[i])

    return neighbour, degree, maxdegree

def load_geodesic_weight(path, name, pointnum):
    
    data = sio.loadmat(path)
    data = data[name]
    
    distance = np.zeros((pointnum, pointnum)).astype('float32')
    distance = data

    return distance

def recover_data(recover_feature, logrmin, logrmax, smin, smax, pointnum):
    logr = recover_feature[:,:,0:3]
    s = recover_feature[:,:,3:9]
    
    resultmax = 0.95
    resultmin = -0.95
    
    s = (smax - smin) * (s - resultmin) / (resultmax - resultmin) + smin
    logr = (logrmax - logrmin) * (logr - resultmin) / (resultmax - resultmin) + logrmin
    
    return s, logr
