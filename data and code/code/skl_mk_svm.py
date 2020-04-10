import numpy as np
from sklearn import svm
from mkl_SVM.combine_kernels import combine_kernels
from sklearn.metrics import accuracy_score
from new_RNA3.mkl.hsic_kernel_weights_norm import hsic_kernel_weights_norm


# RBF kernel function
def kernel_RBF(X, Y, gamma):
    '''
    Python numpy 下的 np.tile有些类似于 matlab 中的 repmat函数。不需要 axis 关键字参数，仅通过第二个参数便可指定在各个轴上的复制倍数。
    :param X: 二维矩阵，类型array;
    :param Y: 二维矩阵，类型array
    :param gamma: 一个数
    :return: 
    '''

    r2 = np.tile(np.sum(X**2, 1), (np.size(Y, 0),1)).T + np.tile(np.sum(Y**2, 1), (np.size(X, 0), 1)) - 2 * np.dot(X, Y.T)
    k = np.exp(-r2 * gamma)
    return k

def skl_mk_svm(train_x,feature_id,train_y,test_x,test_y,c,gamma_list):
    '''
    :param train_x: 训练集特征，类型array
    :param feature_id:类型是list,存放的是特征的索引 
    :param train_y:  训练集标签
    :param test_x: 测试集特征，类型array
    :param test_y: 测试集标签
    :param c: SVM中参数c
    :param gamma_list: SVM中参数g, 类型list
    :return: 
    '''
    predict_y = []
    Scores = []
    kernel_weights = []
    m = int(len(feature_id) / 2)
    num_train_samples = np.size(train_x, 0)
    num_test_samples = np.size(test_x, 0)
    #1.computer training and test kernels (with RBF)
    K_train = []
    K_test = []
    for i in range(m):
        kk_train = kernel_RBF(train_x[:, feature_id[2*i]:feature_id[2*i+1]], train_x[:, feature_id[2*i]:feature_id[2*i+1]], gamma_list[i])
        K_train.append(kk_train)

        kk_test = kernel_RBF(test_x[:, feature_id[2*i]:feature_id[2*i+1]], train_x[:, feature_id[2*i]:feature_id[2*i+1]], gamma_list[i])
        K_test.append(kk_test)

    K_train = np.array(K_train) #这时K_train是三维矩阵了
    K_test = np.array(K_test)  #这时K_test是三维矩阵了

    # 2.multiple kernel learning
    kernel_weights = hsic_kernel_weights_norm(K_train, train_y, 1, 0.1, 0.01)

    # kernel_weights = np.ones((m, 1))
    # kernel_weights = kernel_weights / m
    kernel_weights = kernel_weights.reshape(m,).tolist()

    K_train_com = combine_kernels(kernel_weights, K_train)
    K_test_com = combine_kernels(kernel_weights, K_test)

    K_train_com.tolist()
    K_test_com.tolist()
    # 3.train and test model
    # K_train_com = np.insert(K_train_com, 0, [j for j in range(1, num_train_samples+1)], axis=1)
    # cg_str = ['-t 4 -c '+ str(c)+' -b 1 -q']
    train_y = train_y.reshape(len(K_train_com),).tolist()
    clf = svm.SVC(C=c, kernel='precomputed', probability=True)
    # joblib.dump(clf, 'clf'+str(i)+'.pkl')
    clf.fit(K_train_com, train_y)
    # K_test_com = np.insert(K_test_com, 0, [j for j in range(1, num_test_samples+1)], axis=1)
    y_pred = clf.predict(K_test_com)
    y_score = clf.predict_proba(K_test_com)
    acc = accuracy_score(test_y, y_pred)
    return y_pred, acc, y_score, kernel_weights