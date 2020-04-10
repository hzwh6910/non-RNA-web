import sys
sys.path.extend(["../../", "../", "./"])
from extract_feature import extract_feature, read_fasta
import joblib
from test_miRNA import test_miRNA
import numpy as np
from sklearn import preprocessing
import pandas as pd

def read_fasta(file):
    f = open(file)
    docs = f.readlines()
    fasta = []
    for seq in docs:
        if seq.startswith(">"):
            continue
        else:
            fasta.append(seq)

    return fasta

def miRNA(sequence):
    sequence = [sequence.strip('\n')]
    f1, f2, f3, f4, f5, f6, f7 = extract_feature(sequence)
    # combine matrix
    feature = np.concatenate((f1, f2, f3, f4, f5, f6, f7), axis=1)  # 横向拼接
    gamma_list = [0.5, 0.0625, 0.322916667, 1.104166667, 0.843098958, 0.125, 0.052083333]
    feature_id = [1, 96, 96, 352, 352, 692, 692, 696, 696, 832, 832, 848, 848, 912]
    # train feature
    original_train_feature = pd.read_csv('miRNA_feature.csv', header=None).values
    train_feature = preprocessing.MinMaxScaler().fit_transform(original_train_feature)
    all_feature = np.concatenate((feature, original_train_feature), axis=0)
    all_feature = preprocessing.MinMaxScaler().fit_transform(all_feature)
    feature = np.array([all_feature[0, :]])
    predict = []
    for i in range(6):
        clf = joblib.load('model/miRNA'+'/clf' + str(i + 1) + '.pkl')
        y_pred, y_score, kernel_weights = test_miRNA(train_feature, feature_id, feature, gamma_list, clf)
        y_pred = y_pred.tolist()
        predict.append(y_pred)
    predict = np.array(predict).T
    result = predict
    return result
if __name__ == '__main__':
    sequence = 'CGCGGGAGTTGCTGAGAGGAGACA'
    print(miRNA(sequence))
    exit()
    fastas = read_fasta('miRNA.fasta')
    result = []
    for sequence in fastas:
        res = miRNA(sequence)[0]
        print(res)
        result.append(res)

    # print(np.array(result))
    print(np.array(result).any() == 0)