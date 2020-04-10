from new_RNA3.mkl.skl_mk_svm import skl_mk_svm
import numpy as np
from svmutil import *
from sklearn.model_selection import KFold
from sklearn import preprocessing
import pandas as pd
import matlab
import matlab.engine
eng = matlab.engine.start_matlab()
def evlauate(pre_y, Y_test, pre_score_2):
    average_pre_score = eng.Average_precision(matlab.double(pre_score_2.T.tolist()), matlab.double(Y_test.T.tolist()))
    zero_one_loss_1 = eng.One_error(matlab.double(pre_score_2.T.tolist()), matlab.double(Y_test.T.tolist()))
    coverage_error_1 = eng.coverage(matlab.double(pre_score_2.T.tolist()), matlab.double(Y_test.T.tolist()))
    label_ranking_loss_1 = eng.Ranking_loss(matlab.double(pre_score_2.T.tolist()), matlab.double(Y_test.T.tolist()))
    ham_loss = eng.Hamming_loss(matlab.double(pre_y.T.tolist()), matlab.double(Y_test.T.tolist()))
    acc_score = eng.Accuracy(matlab.double(pre_y.T.tolist()), matlab.double(Y_test.T.tolist()))

    return average_pre_score,  coverage_error_1, label_ranking_loss_1, ham_loss, zero_one_loss_1, acc_score

def model(X_train, Y_train, X_test, Y_test, c, gamma_list):
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    col = np.size(Y_train, 1)
    predict = []
    predict_score = []
    # 每一折的权重
    all_weights1 = np.zeros((1,7))
    all_weights1 = all_weights1[0]
    for i in range(col):
        print(i+1)
        pre_y, acc, pre_score, kernel_weights = skl_mk_svm(X_train, feature_id, Y_train[:, i], X_test, Y_test[:,i], c, gamma_list)
        # print(type(kernel_weights))
        all_weights1 += np.array(kernel_weights)
        pre_y = pre_y.tolist()
        predict.append(pre_y)
        pre_score_temp = pre_score[:, 1]
        pre_score_temp.tolist()
        predict_score.append(pre_score_temp)
    predict = np.array(predict).T
    predict_score = np.array(predict_score).T
    return predict, predict_score, all_weights1
def kfold_cross_validation(feature, feature_label, c, gamma_list):
    kf = KFold(n_splits=10, random_state=True)
    predict_result = []
    predict_result_scores = []
    average_precision_score_list = []
    coverage_error_list = []
    ranking_loss = []
    hamming_loss_list = []
    zero_one_loss_list = []
    acc_score_list = []
    # 所有的权重
    all_weight = np.zeros((1,7))
    all_weight = all_weight[0]
    for train_index, test_index in kf.split(feature):
        print('train_index', train_index, 'test_index', test_index)
        X_train, Y_train = feature[train_index], feature_label[train_index]
        X_test, Y_test = feature[test_index], feature_label[test_index]
        predict, predict_score, all_weights1 = model(X_train, Y_train, X_test, Y_test, c, gamma_list)
        # print(all_weights1)
        all_weight += all_weights1
        predict_result.extend(predict.tolist())
        predict_result_scores.extend(predict_score.tolist())
        average_pre_score, coverage_error_1, label_ranking_loss_1, ham_loss, \
                                                zero_one_loss_1, acc_score = evlauate(predict, Y_test, predict_score)
        average_precision_score_list.append(average_pre_score)
        coverage_error_list.append(coverage_error_1)
        ranking_loss.append(label_ranking_loss_1)
        hamming_loss_list.append(ham_loss)
        zero_one_loss_list.append(zero_one_loss_1)
        acc_score_list.append(acc_score)
    return np.mean(average_precision_score_list), np.mean(coverage_error_list), np.mean(ranking_loss), \
                                            np.mean(hamming_loss_list), np.mean(zero_one_loss_list), \
                                                np.mean(acc_score_list), predict_result, predict_result_scores, all_weight/(np.size(feature_label,1)*10),\
                                            np.std(average_precision_score_list), np.std(acc_score_list),np.std(coverage_error_list),\
                                                np.std(ranking_loss),np.std(hamming_loss_list),np.std(zero_one_loss_list), average_precision_score_list,\
            acc_score_list, coverage_error_list, ranking_loss, hamming_loss_list, zero_one_loss_list


if __name__ == '__main__':
    # load data
    filename = 'human_snoRNA'
    c = 4.5
    f1 = pd.read_csv(r'F:\RNA数据集3\特征\human\\'+filename+'\\'+filename+'_CKSNAP.csv', header=None).values
    f2 = pd.read_csv(r'F:\RNA数据集3\特征\human\\'+filename+'\\'+filename+'_Kmer4.csv', header=None).values
    f3 = pd.read_csv(r'F:\RNA数据集3\特征\human\\'+filename+'\\'+filename+'_Kmer1234.csv', header=None).values
    f4 = pd.read_csv(r'F:\RNA数据集3\特征\human\\'+filename+'\\'+filename+'_NAC.csv', header=None).values
    f5 = pd.read_csv(r'F:\RNA数据集3\特征\human\\'+filename+'\\'+filename+'_RCKmer.csv', header=None).values
    f6 = pd.read_csv(r'F:\RNA数据集3\特征\human\\'+filename+'\\'+filename+'_DNC.csv', header=None).values
    f7 = pd.read_csv(r'F:\RNA数据集3\特征\human\\'+filename+'\\'+filename+'_TNC.csv', header=None).values
    label = pd.read_csv(r'F:\RNA数据集3\选择标签后的数据集与标签_6\\'+filename+'_label.csv', header=None).values
    # combine matrix
    feature = np.concatenate((f1, f2, f3, f4, f5, f6, f7), axis=1)  # 横向拼接
    feature = preprocessing.MinMaxScaler().fit_transform(feature)
    row_dim = np.size(label, 0)
    print(feature.shape)
    print('row_dim=', row_dim)
    print('label.shape=', label.shape)
    if filename=='human_lncRNA':
        # 平均_g
        gamma_list = [1.525,2.05625,1.63125,1.63,1.71875,3.13125,3.26640625]
        #众数
        # gamma_list = [0.125,0.5,1,0.03125,0.03125,0.03125,0.03125]
    elif filename == 'human_snoRNA':
        # 众数_g
        # gamma_list = [0.03125,0.03125,0.03125,0.03125,0.03125,0.03125,0.03125]
        # 平均_g
        gamma_list = [0.0375,0.110416667,0.090625,2.47375,2.072916667,2.065178571,1.8390625]
    elif filename == 'human_miRNA':
        # 众数g
        # gamma_list = [0.03125,0.03125,0.03125,0.03125,0.03125,0.03125,0.03125,0.03125]
        # 平均数g
        gamma_list = [0.244791667,0.440972222,0.408854167,2.728125,2.314236111,2.217261905,1.98046875]
    elif filename == 'human_mRNA':
        # 众数
        # gamma_list = [0.125,0.5,0.25,32,0.5,0.03125,2]
        # 平均数
        gamma_list = [3.775,0.4,0.425,20.85,1.125,1.0063,1.45]
    feature_id = [1,96,96,352,352,692,692,696,696,832,832,848,848,912]
    average_pre_score, coverage_error_1, label_ranking_loss_1, ham_loss, zero_one_loss_1, \
     acc_score, predict_result, predict_result_scores, mean_weight, ap_std,acc_std,cov_std,rl_std,hl_std,zo_std, average_precision_score_list,\
            acc_score_list, coverage_error_list, ranking_loss, hamming_loss_list, zero_one_loss_list = kfold_cross_validation(feature, label, c, gamma_list)
    all_average_precision_score = eng.Average_precision(matlab.double(np.array(predict_result_scores).T.tolist()),
                                                        matlab.double(label.T.tolist()))
    all_accuracy_score = eng.Accuracy(matlab.double(np.array(predict_result).T.tolist()),
                                      matlab.double(label.T.tolist()))
    all_result = []
    every_fold_result = []
    every_fold_result.append(average_precision_score_list)
    every_fold_result.append(acc_score_list)
    every_fold_result.append(coverage_error_list)
    every_fold_result.append(ranking_loss)
    every_fold_result.append(hamming_loss_list)
    every_fold_result.append(zero_one_loss_list)
    # every_fold_result = np.array(every_fold_result)
    df = pd.DataFrame(every_fold_result).T
    df.to_csv('Every_fold_' + filename + '.csv', header=None, index=None)
    print(np.array(predict_result))
    print('all_average_precision_score=',all_average_precision_score)
    print('all_accuracy_score=', all_accuracy_score)
    # print('cov=',coverage_error(label, np.array(predict_result_scores)) - 1)
    # print('RL=',label_ranking_loss(label, np.array(predict_result_scores)))
    # print('all_hamming_loss', hamming_loss(label, np.array(predict_result)))
    # print('all_zero_one_loss=', zero_one_loss(label, np.array(predict_result)))
    all_result.append(all_average_precision_score)
    all_result.append(all_accuracy_score)
    # all_result.append(coverage_error(label, np.array(predict_result_scores)) - 1)
    # all_result.append(label_ranking_loss(label, np.array(predict_result_scores))
    # all_result.append(hamming_loss(label, np.array(predict_result)))
    # all_result.append(zero_one_loss(label, np.array(predict_result)))
    all_result.append(
        eng.coverage(matlab.double(np.array(predict_result_scores).T.tolist()), matlab.double(label.T.tolist())))
    all_result.append(
        eng.Ranking_loss(matlab.double(np.array(predict_result_scores).T.tolist()), matlab.double(label.T.tolist())))
    all_result.append(
        eng.Hamming_loss(matlab.double(np.array(predict_result).T.tolist()), matlab.double(label.T.tolist())))
    all_result.append(
        eng.One_error(matlab.double(np.array(predict_result_scores).T.tolist()), matlab.double(label.T.tolist())))
    print(all_result)
    print('每折平均的情况下：')
    mean_result = []
    mean_std = []
    print('AP=',average_pre_score)
    print('acc=', acc_score)
    print('cov=',coverage_error_1)
    print('RL=',label_ranking_loss_1)
    print('hamming_loss=', ham_loss)
    print('zero_one_loss', zero_one_loss_1)
    mean_result.append(average_pre_score)
    mean_result.append(acc_score)
    mean_result.append(coverage_error_1)
    mean_result.append(label_ranking_loss_1)
    mean_result.append(ham_loss)
    mean_result.append(zero_one_loss_1)
    print('平均值：', end='\n')
    print(mean_result)
    mean_std.append(ap_std)
    mean_std.append(acc_std)
    mean_std.append(cov_std)
    mean_std.append(rl_std)
    mean_std.append(hl_std)
    mean_std.append(zo_std)
    print('标准差：',end='\n')
    print(mean_std)
    print('平均权重=')
    for i in mean_weight:
        print(i, end=',')
    # pred_score = np.array(predict_result_scores)
    # pred_label = predict_result
    # Y_true = label
    # df1 = pd.DataFrame(pred_score)
    # df2 = pd.DataFrame(pred_label)
    # df3 = pd.DataFrame(Y_true)
    # df1.to_csv(filename + '_pred_score.csv', header=None, index=None)
    # df2.to_csv(filename + '_pred_label.csv', header=None, index=None)
    # df3.to_csv(filename + '_Y_true.csv', header=None, index=None)

