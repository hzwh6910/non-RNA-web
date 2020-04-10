import numpy as np
from svmutil import *
from scipy.optimize import minimize
##----------------------------------------------------------------------------------------------------------------------##
def kernel_RBF(X, Y):
    '''
    Python numpy 下的 np.tile有些类似于 matlab 中的 repmat函数。不需要 axis 关键字参数，仅通过第二个参数便可指定在各个轴上的复制倍数。
    :param X: 二维矩阵，类型array;
    :param Y: 二维矩阵，类型array
    :param gamma: 一个数
    :return: 
    '''

    k = np.tile(np.sum(X**2, 1), (np.size(Y, 0),1)).T + np.tile(np.sum(Y**2, 1), (np.size(X, 0), 1)) - 2 * np.dot(X, Y.T)
    return k

def label_matrix(label):
    ylabel = []
    for i in label:
        if i == 1:
            ylabel.append([1, 0, 0, 0, 0, 0, 0, 0])
        elif i == 2:
            ylabel.append([0, 1, 0, 0, 0, 0, 0, 0])
        elif i == 3:
            ylabel.append([0, 0, 1, 0, 0, 0, 0, 0])
        elif i == 4:
            ylabel.append([0, 0, 0, 1, 0, 0, 0, 0])
        elif i == 5:
            ylabel.append([0, 0, 0, 0, 1, 0, 0, 0])
        elif i == 6:
            ylabel.append([0, 0, 0, 0, 0, 1, 0, 0])
        elif i == 7:
            ylabel.append([0, 0, 0, 0, 0, 0, 1, 0])
        elif i == 8:
            ylabel.append([0, 0, 0, 0, 0, 0, 0, 1])
    ylabel = np.array(ylabel)
    for i in range(len(ylabel)):
        for j in range(len(ylabel[0,:])):
            if ylabel[i][j] == 0:
                ylabel[i][j] = -1
    return ylabel
##------------------------------------------------------------------------------------------------------------##
def obj_function(w,Mi,ai,regcoef1,regcoef2):
    J = -1*np.dot(w.T, ai) + regcoef1*np.dot(np.dot(w.T, Mi), w) + regcoef2*(np.linalg.norm(w, ord=2, keepdims=True))**2
    return J
def con(args):
    cons = ({'type': 'eq', 'fun': lambda w:sum(w)-1}, {'type': 'ineq', 'fun': lambda w:w[0]},
            {'type': 'ineq', 'fun': lambda w:w[1]},{'type': 'ineq', 'fun': lambda w:w[2]},
            {'type': 'ineq', 'fun': lambda w:w[3]},{'type': 'ineq', 'fun': lambda w:w[4]},
            {'type': 'ineq', 'fun': lambda w: w[5]},{'type': 'ineq', 'fun': lambda w:w[6]},
            {'type': 'ineq', 'fun': lambda w:1-w[0]},{'type': 'ineq', 'fun': lambda w:1-w[1]},
            {'type': 'ineq', 'fun': lambda w:1-w[2]},{'type': 'ineq', 'fun': lambda w:1-w[3]},
            {'type': 'ineq', 'fun': lambda w:1-w[4]},{'type': 'ineq', 'fun': lambda w:1-w[5]},
            {'type': 'ineq', 'fun': lambda w: 1 - w[6]}
            )
    return cons


def optimize_weights(x0, fun):
    '''
    :param x0: 
    :param fun: 
    :return: 
    '''
    n = np.size(x0, 0)
    Aineq = []
    bineq = []
    Aeq = np.ones((1, n))
    beq = 1
    LB = np.zeros((1, n))
    UB = np.ones((1, n))
    args = ()
    cons = con(args)
    res = minimize(fun, x0, method='SLSQP', constraints=cons)
    # res = minimize(fun, x0, method='SLSQP')
    # print(res.x)
    return res.x



def hsic_kernel_weights_norm(Kernels_list, adjmat,dim, regcoef1, regcoef2):
    '''
    :param Kernels_list: K_train 训练集， 类型array
    :param adjmat:  train_y 训练标签， 类型list
    :param dim:  1
    :param regcoef1: 0.01 
    :param regcoef2: 0.001
    :return: 
    '''
    # adjmat = np.array(adjmat)
    num_samples = np.size(adjmat, 0)
    num_kernels = np.size(Kernels_list, 0)
    weight_v = np.zeros((num_kernels, 1))
    y = adjmat.reshape(num_samples, 1)
    # y = label_matrix(y) # 实现标签的转化，将一列标签转化为多列标签；
    # y = preprocessing.MinMaxScaler().fit_transform(y)
    # Graph based kernel
    if dim == 1:
        ideal_kernel = np.dot(y, y.T)
    else:
        ideal_kernel = np.dot(y.T, y)
    # ideal_kernel = kernel_RBF(ideal_kernel, ideal_kernel) # 求标签之间的相似度矩阵
    # ideal_kernel=Knormalized(ideal_kernel)
    # print(ideal_kernel.shape)
    N_U = np.size(ideal_kernel, 0)
    l = np.ones((N_U, 1))
    H = np.eye(N_U, dtype=float) - np.dot(l, l.T)/N_U # H:NxN的矩阵

    M = np.zeros((num_kernels, num_kernels))
    for i in range(num_kernels):
        for j in range(num_kernels):
            kk1 = np.dot(np.dot(H, Kernels_list[i, :, :]), H)
            kk2 = np.dot(np.dot(H, Kernels_list[j, :, :]), H)
            mm = np.trace(np.dot(kk1.transpose(), kk2))
            m1 = np.trace(np.dot(kk1, kk1.transpose()))
            m2 = np.trace(np.dot(kk2, kk2.transpose()))
            M[i, j] = mm / (np.sqrt(m1) * np.sqrt(m2))
    d_1 = sum(M)
    D_1 = np.diag(d_1)
    LapM = D_1 - M

    a = np.zeros((num_kernels, 1))
    for i in range(num_kernels):
        kk = np.dot(np.dot(H, Kernels_list[i, :, :]), H)
        aa = np.trace(np.dot(kk.transpose(), ideal_kernel))
        a[i] = aa*((N_U-1)**-2)

    v = np.random.rand(num_kernels, 1)
    falpha = lambda v:obj_function(v, LapM, a, regcoef1, regcoef2)
    x_alpha = optimize_weights(v, falpha)
    weight_v = x_alpha
    return weight_v
