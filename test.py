import numpy as np
import os
from sklearn.model_selection import KFold
import random
from sklearn.svm import SVC
import warnings
import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.linear_model import Lasso
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


def sigmoid_y(x, thresold=0.5):
    if x < thresold:
        x = 0
    else:
        x = 1
    return x


def getAccSenSpcAuc(label, pre, pre_bestthresold=None):
    """
    只适用于二分类
    :param label:01标签
    :param pre:属于1类的概率
    :param pre_bestthresold:可手动设置阈值，否则返回roc曲线上约登指数最大处阈值
    :return:
    """
    final_true_label = label
    final_pred_value = pre
    patient_num = len(final_true_label)

    # 计算auc，并计算最佳阈值
    if (sum(final_true_label) == patient_num) or (sum(final_true_label) == 0):
        Aucc = 0
        print('only one class')
    else:
        Aucc = metrics.roc_auc_score(final_true_label, final_pred_value)
        # print('AUC', Aucc)

    # 计算最佳阈值
    fpr, tpr, thresholds = metrics.roc_curve(final_true_label, final_pred_value)
    # 计算约登指数
    Youden_index = tpr + (1 - fpr)
    best_thresold = thresholds[Youden_index == np.max(Youden_index)][0]

    if best_thresold > 1:
        best_thresold = 0.5

    # 如果有预设阈值，则使用预设阈值计算acc，sen，spc
    if pre_bestthresold is not None:
        best_thresold = pre_bestthresold

    # 根据最终list来计算最终指标
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    pred = []
    for nn in range(patient_num):
        t_label = final_true_label[nn]  # true label
        p_value = final_pred_value[nn]

        p_label = sigmoid_y(p_value, best_thresold)
        pred.append(p_label)
        if (t_label == 1) and (t_label == p_label):
            tp = tp + 1  # 真阳
        elif (t_label == 0) and (t_label == p_label):
            tn = tn + 1  # 真阴
        elif (t_label == 1) and (p_label == 0):
            fn = fn + 1  # 假阴
        elif (t_label == 0) and (p_label == 1):
            fp = fp + 1  # 假阳

    Sensitivity = tp / ((tp + fn) + (1e-16))
    Specificity = tn / ((tn + fp) + (1e-16))
    Accuracy = (tp + tn) / ((tp + tn + fp + fn) + (1e-16))

    return [Accuracy, Sensitivity, Specificity, Aucc, best_thresold]


UseSVM = True  # 是否使用SVM

# 加载数据，分为三部分：训练集、测试集1、测试集2

data_in_0 = np.load('')
label_in = np.load('')
data_out_0 = np.load('')
label_out = np.load('')
data_independent_0 = np.load('')
label_independent = np.load('')

selected_feature_index = np.load('./selected_feature_index.npy')   # 特征选择后的特征序列
data_in = data_in_0[:, selected_feature_index]
data_out = data_out_0[:, selected_feature_index]
data_independent = data_independent_0[:, selected_feature_index]



# 下面部分为常规的训练测试过程
case_num = len(label_in)
sfolder = KFold(n_splits=case_num, random_state=2021, shuffle=True)

svm_C = [0.95]
svm_Gamma = [6.8,]
svm_max_iter = [28]

for final_svm_C in svm_C:
    for final_svm_Gamma in svm_Gamma:
        for final_svm_max_iter in svm_max_iter:
            prob_svm = []
            label_CV = []
            id_CV = []

            label_CV_out = []
            prob_svm_out = []

            label_CV_id = []
            prob_svm_id = []

            for train, test in sfolder.split(data_in, label_in):
                data_train = data_in[train, :]
                label_train = label_in[train]
                data_test = data_in[test, :]
                label_test = label_in[test]
                id_CV = id_CV + list(test)

                StandardScaler = preprocessing.MinMaxScaler()
                data_train = StandardScaler.fit_transform(data_train)
                data_test = StandardScaler.transform(data_test)
                data_out_ = StandardScaler.transform(data_out)
                data_independent_ = StandardScaler.transform(data_independent)

                clf = SVC(probability=True,
                          kernel='rbf',
                          C=final_svm_C,
                          gamma=final_svm_Gamma,
                          class_weight={0: 1, 1: 1},
                          max_iter=final_svm_max_iter, random_state=2021)

                clf.fit(data_train, label_train)
                y_prob = clf.predict_proba(data_test)
                prob_svm = prob_svm + list(y_prob[:, 1])

                y_prob = clf.predict_proba(data_out_)
                prob_svm_out.append(list(y_prob[:, 1]))

                y_prob = clf.predict_proba(data_independent_)
                prob_svm_id.append(list(y_prob[:, 1]))
                label_CV = label_CV + list(label_test)


            acc_svm, sen_svm, spc_svm, AUC_svm, bst_svm = getAccSenSpcAuc(label_CV, prob_svm)
            print(acc_svm, sen_svm, spc_svm, AUC_svm)

            label_CV_out = list(label_out)
            label_CV_id = list(label_independent)

            prob_svm_out = list(np.mean(np.array(prob_svm_out), axis=0))
            acc_svm_out, sen_svm_out, spc_svm_out, AUC_svm_out, bst_svm_out = getAccSenSpcAuc(label_CV_out, prob_svm_out)
            print(acc_svm_out, sen_svm_out, spc_svm_out, AUC_svm_out)

            prob_svm_id = list(np.mean(np.array(prob_svm_id), axis=0))
            acc_svm_id, sen_svm_id, spc_svm_id, AUC_svm_id, bst_svm_id = getAccSenSpcAuc(label_CV_id, prob_svm_id)
            print(acc_svm_id, sen_svm_id, spc_svm_id, AUC_svm_id)

