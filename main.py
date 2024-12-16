# Importing the libraries
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,roc_auc_score
import matplotlib.pyplot as plt
from model import GAE
from config import get_args

from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, matthews_corrcoef
from sklearn.utils import shuffle

args = get_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
drug_embeddings = np.load('drug_embeddings.npy')
# 加载 protein_embeddings 文件
protein_embeddings = np.load('protein_embeddings.npy')
p_features_side = torch.from_numpy(protein_embeddings).to(device)  # 988,635
d_features_side = torch.from_numpy(drug_embeddings).to(device)  # 791,635
nums_protein = 1512
nums_drug = 708

# 训练函数
def train_model(X_train, y_train):
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64, 16),
        max_iter=1000,
        activation='relu',
        solver='sgd',
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=100,
        alpha=0.01
    )
    clf.fit(X_train, y_train)
    return clf
import lightgbm as lgb

# def train_model(X_train, y_train):
#     clf = lgb.LGBMClassifier(
#         n_estimators=500,         # 树的数量
#         learning_rate=0.01,        # 学习率
#         max_depth=7,              # 最大树深度
#         objective='binary',       # 二分类
#         random_state=2024
#     )
#     clf.fit(X_train, y_train)
#     return clf
# from catboost import CatBoostClassifier
#
# def train_model(X_train, y_train):
#     clf = CatBoostClassifier(
#         iterations=500,           # 迭代次数
#         learning_rate=0.02,        # 学习率
#         depth=6,                   # 树的深度
#         loss_function='Logloss',   # 使用Logloss作为损失函数
#         verbose=0,                 # 关闭训练输出信息
#         random_state=2024
#     )
#     clf.fit(X_train, y_train)
#     return clf
import xgboost as xgb

# def train_model(X_train, y_train):
#     clf = xgb.XGBClassifier(
#         n_estimators=100,         # 树的数量
#         learning_rate=0.05,        # 学习率
#         max_depth=6,              # 最大树深度
#         objective='binary:logistic',  # 二分类任务
#         use_label_encoder=False,  # 不使用标签编码器
#         eval_metric='logloss',    # 使用log损失作为评估指标
#         random_state=2024
#     )
#     clf.fit(X_train, y_train)
#     return clf


def evaluate_model(clf, X_test, y_test):
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    return accuracy, auc, precision, recall, f1, mcc, specificity

def load_data():
    pairs_df = pd.read_pickle("pairs.pkl")
    pairs_df = pd.DataFrame(pairs_df)
    pairs = pairs_df.iloc[:, :2].values
    labels = pairs_df.iloc[:, 1].values
    return pairs, labels

# 主训练循环
def main():
    pairs, labels = load_data()
    labels = torch.tensor(labels).float()
    model = GAE(nums_protein, nums_drug,
                 p_features_side, d_features_side,
                 args.emb_dim, args.hidden,
                args.dropout, args.attention_dropout_rate, args.slope,
                args.num_heads)
    model.to(device)

    features = model()

    labels = labels.cpu().numpy()  # 转换为NumPy数组
    indices = np.arange(len(features))
    indices, features, labels = shuffle(indices, features, labels, random_state=2024)

    kf = KFold(n_splits=5, shuffle=True, random_state=2024)
    scores, aucs, precisions, recalls, f1s, mccs, specs = [], [], [], [], [], [], []
    mean_fpr = np.linspace(0, 1, 5000)
    tprs = []
    mean_recall = np.linspace(0, 1, 5000)
    precisions_all = []

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 7))

    for i, (train_index, val_index) in enumerate(kf.split(features)):
        X_train, X_val = features[train_index], features[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        # 训练模型
        clf = train_model(X_train, y_train)

        # 评估模型
        val_accuracy, val_auc, val_precision, val_recall, val_f1, val_mcc, val_spec = evaluate_model(clf, X_val, y_val)
        scores.append(val_accuracy)
        aucs.append(val_auc)
        precisions.append(val_precision)
        recalls.append(val_recall)
        f1s.append(val_f1)
        mccs.append(val_mcc)
        specs.append(val_spec)

        print(
            f"Fold {i + 1}: val_accuracy: {val_accuracy:.4f}, val_auc: {val_auc:.4f}, val_precision: {val_precision:.4f}, val_recall: {val_recall:.4f}, val_f1 Score: {val_f1:.4f}, Val_MCC: {val_mcc:.4f}, Val_Specificity: {val_spec:.4f}")

        # 绘制ROC曲线
        fpr, tpr, _ = roc_curve(y_val, clf.predict_proba(X_val)[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, alpha=0.4, linestyle='--', label=f'ROC fold {i + 1} (AUC = {val_auc:.4f})')

        # 绘制Precision-Recall曲线
        precision, recall, _ = metrics.precision_recall_curve(y_val, clf.predict_proba(X_val)[:, 1])
        precisions_all.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
        precisions_all[-1][0] = 1.0  # 确保 recall=0 时 precision=1
        pr_auc = auc(recall, precision)
        ax_pr.plot(recall, precision, alpha=0.4, linestyle='--', label=f'PR fold {i + 1} (AUC = {pr_auc:.4f})')

    # 计算和绘制平均ROC曲线
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax_roc.plot(mean_fpr, mean_tpr, color='BlueViolet', label=f'Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})',
                alpha=0.9)
    ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.2)
    ax_roc.set(title="Receiver Operating Characteristic", xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax_roc.set_xlim(left=-0.05, right=1.05)
    ax_roc.set_ylim(bottom=-0.05, top=1.05)
    ax_roc.legend(loc="lower right")

    # 计算和绘制平均Precision-Recall曲线
    mean_precision = np.mean(precisions_all, axis=0)
    mean_precision = np.append(mean_precision, 0.0)  # 确保 recall=1 时 precision=0
    mean_recall = np.append(mean_recall, 1.0)  # 添加 recall=1 的点
    mean_pr_auc = auc(mean_recall, mean_precision)
    std_pr_auc = np.std(precisions_all)
    ax_pr.plot(mean_recall, mean_precision, color='BlueViolet',
               label=f'Mean PR (AUC = {mean_pr_auc:.4f} ± {std_pr_auc:.4f})', alpha=0.9)
    # 添加虚线
    ax_pr.plot([0, 1], [1, 0], linestyle='--', color='gray', alpha=0.7)

    ax_pr.set(title="Precision-Recall Curve", xlabel='Recall', ylabel='Precision')
    ax_pr.set_xlim(left=-0.05, right=1.05)
    ax_pr.set_ylim(bottom=-0.05, top=1.05)
    ax_pr.legend(loc="lower left")

    plt.show()

    # 输出平均结果
    print(f"Model average accuracy: {np.mean(scores):.4f}, AUC: {np.mean(aucs):.4f}")
    print(f"Model average precision: {np.mean(precisions):.4f}, recall: {np.mean(recalls):.4f}")
    print(f"Model average F1 score: {np.mean(f1s):.4f}, MCC: {np.mean(mccs):.4f}, Specificity: {np.mean(specs):.4f}")
if __name__ == '__main__':
    main()
