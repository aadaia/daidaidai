import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from model import *
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads=8, d_ff=None, dropout=0.1,
                 activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x, attn_mask=None, length_mask=None):
        N = x.shape[0]
        L = x.shape[1]

        attn_output, _ = self.attention(
            x, x, x,
            key_padding_mask=attn_mask
        )

        x = x + self.dropout(attn_output)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x + y)
def DecomAdjMat(adj_mat, weight_mat, lambda_0, lambda_1, max_iter_num, k, thre_val):

    row_num, col_num = adj_mat.shape
    A_tran = adj_mat.transpose()

    A_A_tran = np.matmul(adj_mat, A_tran)

    W_had_W = np.multiply(weight_mat, weight_mat)
    W2_had_A = np.multiply(W_had_W, adj_mat)

    #### initialize U,V,C ###################################
    rng = np.random.default_rng(seed=42)
    U = np.abs(rng.normal(loc=0, scale=1, size=(row_num, k)))
    V = np.abs(rng.normal(loc=0, scale=1, size=(row_num, k)))

    C0 = np.abs(rng.normal(loc=0, scale=1, size=(k, k)))
    C = (C0 + C0.transpose()) / 2
    ##########################################################

    i = 1
    while i <= max_iter_num:

        ######################## start update U ######################################

        W2_had_A_V = np.matmul(W2_had_A, V)

        A_tran_U = np.matmul(A_tran.A, U)
        A_A_tran_U = np.matmul(adj_mat.A, A_tran_U)

        C_tran_add_C = C.transpose() + C

        A_A_tran_U_C_tran_add_C = np.matmul(A_A_tran_U, C_tran_add_C)

        U_lr_numerator = 1 * W2_had_A_V + 1 * A_A_tran_U_C_tran_add_C

        U_V_tran = np.matmul(U, V.transpose())
        W2_had_U_V_tran = np.multiply(W_had_W, U_V_tran)

        W2_had_U_V_tran_V = np.matmul(W2_had_U_V_tran, V)

        U_tran_U = np.matmul(U.transpose(), U)
        U_C = np.matmul(U, C)
        U_C_tran = np.matmul(U, C.transpose())

        U_C_U_tran_U = np.matmul(U_C, U_tran_U)
        U_C_U_tran_U_C_tran = np.matmul(U_C_U_tran_U, C.transpose())

        U_C_tran_U_tran_U = np.matmul(U_C_tran, U_tran_U)
        U_C_tran_U_tran_U_C = np.matmul(U_C_tran_U_tran_U, C)

        U_lr_denominator = 1 * W2_had_U_V_tran_V + 1 * (U_C_U_tran_U_C_tran + U_C_tran_U_tran_U_C) + lambda_0 * U
        U_lr_denominator = np.maximum(U_lr_denominator, np.where(U_lr_denominator < 1.0e-16, 1, 0) * thre_val)

        U = (U_lr_numerator / U_lr_denominator) * U
        ######################### finish update U ######################################################################
        ##################################### start update V #########################################

        W2_had_A_tran = W2_had_A.transpose()
        W2_had_A_tran_U = np.matmul(W2_had_A_tran, U)

        V_lr_numerator = 1 * W2_had_A_tran_U

        U_V_tran_update = np.matmul(U, V.transpose())
        W2_had_U_V_tran_update = np.multiply(W_had_W, U_V_tran_update)
        W2_had_U_V_tran_update_tran = W2_had_U_V_tran_update.transpose()

        W2_had_U_V_tran_update_tran_U = np.matmul(W2_had_U_V_tran_update_tran, U)

        V_lr_denominator = 1 * W2_had_U_V_tran_update_tran_U + lambda_1 * V
        V_lr_denominator = np.maximum(V_lr_denominator, np.where(V_lr_denominator < 1.0e-16, 1, 0) * thre_val)

        V = (V_lr_numerator / V_lr_denominator) * V
        ##################################### finish update V ########################################
        ##################################### start update C ########################################

        A_tran_U = np.matmul(A_tran.A, U)
        U_tran_A_A_tran_U = np.matmul(A_tran_U.transpose(), A_tran_U)

        C_lr_numerator = 1 * U_tran_A_A_tran_U + 1 * C.transpose()

        U_tran_U = np.matmul(U.transpose(), U)
        U_tran_U_C = np.matmul(U_tran_U, C)
        U_tran_U_C_U_tran_U = np.matmul(U_tran_U_C, U_tran_U)

        C_lr_denominator = 1 * U_tran_U_C_U_tran_U + 1 * C
        C_lr_denominator = np.maximum(C_lr_denominator, np.where(C_lr_denominator < 1.0e-16, 1, 0) * thre_val)

        C = (C_lr_numerator / C_lr_denominator) * C
        ###################################### finish update C ######################################
        ###################################### compute loss function ################################

        U_update_V_tran_update = np.matmul(U, V.transpose())
        A_minus_U_update_V_tran_update = adj_mat.A - U_update_V_tran_update
        W_A_minus_U_update_V_tran_update = np.multiply(weight_mat, A_minus_U_update_V_tran_update)
        L_A = np.linalg.norm(W_A_minus_U_update_V_tran_update, 'fro') ** 2

        U_C_update = np.matmul(U, C)
        U_C_U_tran_update = np.matmul(U_C_update, U.transpose())
        A_A_tran_minus_U_C_U_tran_update = A_A_tran - U_C_U_tran_update
        L_A_A_tran = np.linalg.norm(A_A_tran_minus_U_C_U_tran_update, 'fro') ** 2

        C_minus_C_tran_update = C - C.transpose()
        L_C_C_tran = np.linalg.norm(C_minus_C_tran_update, 'fro') ** 2

        L_U = np.linalg.norm(U, 'fro') ** 2
        L_V = np.linalg.norm(V, 'fro') ** 2

        L = 1 / 2 * L_A + 1 / 2 * L_A_A_tran + 1 / 4 * L_C_C_tran + lambda_0 / 2 * L_U + lambda_1 / 2 * L_V
        print("The number of iterations is " + str(i) + "," + "loss value is " + str(L))
        ####################################### finish compute loss function ########################
        if i == max_iter_num or L < thre_val:
            return U, V, C
        else:
            i += 1

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)


        return x
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x




