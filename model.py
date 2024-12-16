import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
# from torchvision import models
import sklearn
import random
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score
from layers import *

# from mxnet.gluon import loss as gloss
from sklearn.metrics import precision_recall_curve
from  sklearn import metrics
from sklearn.decomposition import NMF
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GAE(nn.Module): # 图自编码
    def __init__(self, num_proteins, num_drugs,
                        p_features_side, d_features_side,
                        emb_dim, hidden, dropout, attention_dropout_rate,slope,num_heads, **kwargs):
        super(GAE, self).__init__()

        self.num_proteins = num_proteins # 6040
        self.num_drugs = num_drugs # 3706
        self.dropout = dropout # 0.7
        self.slope = slope
        self.attention_dropout_rate = attention_dropout_rate

        self.num_heads = num_heads

        self.p_features_side = p_features_side  # 6040,3487
        self.d_features_side = d_features_side  # 3706,3487
        # self.encoders = EncoderLayer(hidden[0], hidden[1], dropout, attention_dropout_rate, num_heads)
        self.nfm = NMF(n_components=128, init='random', random_state=1234)
        self.encoders = EncoderLayer(hidden[0], hidden[1], dropout, attention_dropout_rate, num_heads)
        self.denseu1 = nn.Linear(200, emb_dim, bias=True)  # 3487 ，32 用于设置网络中的全连接层的，需要注意的是全连接层的输入与输出都是二维张量
        self.densev1 = nn.Linear(100, emb_dim, bias=True)  # 3487 ，32
        self.denseu2 = nn.Linear(96832, hidden[2], bias=False)  # 32+32,16
        self.densev2 = nn.Linear(45376, hidden[2], bias=False)  # 32+32,16

    def normalize(self, mx):
        rowsum = torch.sum(mx, 0)
        r_inv = torch.pow(rowsum, -0.5)
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        colsum = torch.sum(mx, 1)
        c_inv = torch.pow(colsum, -0.5)
        c_inv[torch.isinf(c_inv)] = 0.
        c_mat_inv = torch.diag(c_inv)
        mx = torch.matmul(mx, r_mat_inv)
        mx = torch.matmul(c_mat_inv, mx)
        return mx

    def forward(self):
        lambda_0 = 0.1
        lambda_1 = 0.1
        max_iter_num = 2000
        k = 512
        thre_val = 1.0e-8

        # Spa_mat_update = torch.tensor(np.loadtxt("E:\mfdti\luo_data.txt"), dtype=torch.float32)
        # # Spa_mat_update = Spa_mat + Spa_mat.t()
        # # train_data, test_data = train_test_split(Spa_mat_update)
        # col_l2_norm = np.linalg.norm(Spa_mat_update, axis=0)
        # col_l2_norm_vec2mat = np.tile(col_l2_norm, (Spa_mat_update.shape[0], 1))
        # col_normalization = np.nan_to_num(Spa_mat_update / col_l2_norm_vec2mat)
        # # 稀疏矩阵乘法
        # weight_mat = col_normalization.transpose() @ col_normalization  # 或者使用 .dot()
        # weight_mat = np.array(weight_mat)
        # # Spa_mat_update = torch.sparse_coo_tensor(torch.stack(indices), values, Y.shape, dtype=Y.dtype)
        #
        # A, B, C = DecomAdjMat(col_normalization, weight_mat, lambda_0, lambda_1, max_iter_num, k, thre_val)

        Y = torch.tensor(np.loadtxt("D:\MicrosoftEage\DTINet-master (1)\DTINet-master\data\mat_drug_protein.txt"), dtype=torch.float32)
        Y = torch.tensor(np.loadtxt("E:\mfdti\dpinteraction.txt"), dtype=torch.float32)
        Y = pd.read_csv("E:\jing-IC - 副本 - 副本 (2)分类器版\protein_drug interaction.csv",header= 0,encoding='utf-8')
        A = self.nfm.fit_transform(Y)  # Basis matrix
        B = self.nfm.components_
        B = B.T
        #拿pf和df当残差试试
        p_f = self.denseu1(self.p_features_side)

        d_f = self.densev1(self.d_features_side)

        p_h = self.encoders(p_f)
        d_h = self.encoders(d_f)
        p_h = p_h.view(1513, -1)
        p_h = self.denseu2(p_h)
        d_h = d_h.view(709, -1)
        d_h = self.densev2(d_h)
        d_z = F.dropout(d_h, self.dropout).cpu()

        p_z = F.dropout(p_h, self.dropout).cpu()
        p_z= p_z[:-1, :]
        d_z = d_z[:-1, :]
        # d_z = d_z + self.d_features_side
        # p_z = p_z + self.p_features_side
        d_a = np.concatenate((d_z.detach().numpy(), A), axis=1)

        p_b = np.concatenate((p_z.detach().numpy(), B), axis=1)
        #
        # d_a =np.concatenate(d_a , self.d_features_side)
        # p_b = np.concatenate(p_b , self.p_features_side)## DNN
        pairs_df = np.load('df_renamed.npy')
        pairs_df = pd.DataFrame(pairs_df)

        # 提取特征
        pairs = pairs_df.iloc[:, :2].values  # 假设前两列是药物和靶标的索引

        # features = np.array([np.concatenate((p_b[p], d_a[d])) for p, d in pairs])
        # features = np.empty((len(pairs), p_b.shape[1] + d_a.shape[1]))
        features = np.empty((len(pairs), p_b.shape[1] + d_a.shape[1]))
        # features = np.empty((len(pairs), p_z.shape[1] + d_z.shape[1]))

        # 遍历 pairs 中的每对索引 (p, d)
        for i, (d, p) in enumerate(pairs):
            p_feature = p_b[p]
            d_feature = d_a[d]
            # p_feature = p_z[p].detach().numpy()
            # d_feature = d_z[d].detach().numpy()
            features[i] = np.concatenate((p_feature, d_feature))
            # 将 p_b 中索引 p 的药物特征和 d_a 中索引 d 的蛋白质特征连接起来，形成一个特征向量

        return features
