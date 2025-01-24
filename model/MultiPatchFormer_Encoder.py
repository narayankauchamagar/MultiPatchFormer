import torch
import torch.nn as nn
import torch.nn.functional as F
#from layers import gpt2
import numpy as np
from torch.nn import Module
from torch.utils.data import TensorDataset, DataLoader, Dataset
import math
from torch.nn.modules import ModuleList, Module
import torch.nn.functional as F
from einops import rearrange, repeat
#from statsmodels.tsa.seasonal import STL
from typing import List, Tuple
#from layers.Attn_Embedding_ import  AttentionEmbedding_



class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        Args:
            num_features: the number of features or channels
            eps: a value added for numerical stability
            affine: if True, RevIN has learnable affine parameters
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x: torch.Tensor):
        """
        x: (B, L, N)
        """
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class MultiHeadAttention(Module):
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str,
                 mask: bool=False,
                 dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.W_q = torch.nn.Linear(d_model, q * h)
        self.W_k = torch.nn.Linear(d_model, q * h)
        self.W_v = torch.nn.Linear(d_model, v * h)
        self.W_o = torch.nn.Linear(v * h, d_model)
        #self.W_oh = torch.nn.Linear(v * h, d_model)
        #self.W_ohh = torch.nn.Linear(v * h, d_model)
        self.device = device
        self._h = h
        self._q = q
        self.mask = mask
        self.dropout = torch.nn.Dropout(p=dropout)
        #self.norm = torch.nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        #self.conv = torch.nn.Conv1d(in_channels=L, out_channels=rank, kernel_size=3, stride=3, padding=1)
        self.score = None        

    def forward(self, x):

        B, L, D = x.shape
        #x_r = self.conv(x.permute(0, 2, 1)).transpose(1, 2)
        
        #Q = torch.cat(self.W_q(x).chunk(self._h, dim=-1), dim=0)
        #K = torch.cat(self.W_k(x).chunk(self._h, dim=-1), dim=0)
        #V = torch.cat(self.W_v(x).chunk(self._h, dim=-1), dim=0)
        Q = self.W_q(x).reshape(B, L, self._h, self._q).permute(0, 2, 1, 3) #+ self.pe
        K = self.W_k(x).reshape(B, L, self._h, self._q).permute(0, 2, 1, 3) #+ self.pe
        V = self.W_v(x).reshape(B, L, self._h, self._q).permute(0, 2, 1, 3) #+ self.pe


        scores = torch.einsum('bhld, bhjd -> bhlj', Q, K)
        #scores_dh = torch.einsum('bhld, bhle -> bhde', Q, K)
        ##scores_h = torch.einsum('bhld, bald -> bha', Q, K)
        #score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self._q)
        #score = torch.matmul(Q - K, (Q - K).transpose(-1, -2))
        if self.mask:
           mask = torch.ones_like(scores[0]) #self.rel_pos.to(self.device) #torch.ones_like(scores[0])
           mask = torch.tril(mask, diagonal=0)
           scores = torch.where(mask > 0, scores, torch.Tensor([-2**32+1]).expand_as(scores[0]).to(self.device))

        # if self.mask:
        #      mask_dh = torch.ones_like(scores_dh[0]) #self.rel_pos.to(self.device) #torch.ones_like(scores[0])
        #      mask_dh = torch.tril(mask_dh, diagonal=0)
        #      scores_dh = torch.where(mask_dh > 0, scores_dh, torch.Tensor([-2**32+1]).expand_as(scores_dh[0]).to(self.device))
            

        scores = F.softmax(scores / math.sqrt(self._q), dim=-1) 
        self.score = scores
       # scores_dh = F.softmax(scores_dh / math.sqrt(L), dim=-1) 
        ##scores_h = F.softmax(scores_h / math.sqrt(L*self._q), dim=-1) 
        #attention = torch.matmul(score, V) #.transpose(-1, -2))
        atten_out = torch.einsum('bhlj, bhjd -> bhld', scores, V)
        #atten_out_dh = torch.einsum('bhde, bhle -> bhld', scores_dh, V)
        ##atten_out_h = torch.einsum('bha, bald -> bhld', scores_h, V)

        self_attention = self.W_o(atten_out.permute(0, 2, 3, 1).reshape(B, L, -1)) #+ self.pe
        #self_attention_dh = self.W_oh(atten_out_dh.permute(0, 2, 3, 1).reshape(B, L, -1)) #+ self.pe
        ##self_attention_h = self.W_ohh(atten_out_h.permute(0, 2, 3, 1).reshape(B, L, -1))
        #attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        return self_attention , self.score 


#####################################################
class MultiHeadAttention_ch(Module):
    def __init__(self,
 
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str,
                 mask: bool=False,
                 dropout: float = 0.1):
        super(MultiHeadAttention_ch, self).__init__()

        self.W_q = torch.nn.Linear(d_model, q * h)
        self.W_k = torch.nn.Linear(d_model, q * h)
        self.W_v = torch.nn.Linear(d_model, v * h)
        self.W_o = torch.nn.Linear(v * h, d_model)
        #self.W_oh = torch.nn.Linear(v * h, d_model)
        ###self.W_ohh = torch.nn.Linear(v * h, d_model)
        self.device = device
        self._h = h
        self._q = q
        self.mask = mask
        self.dropout = torch.nn.Dropout(p=dropout)
        #self.norm = torch.nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        self.conv = torch.nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1, stride=1,
                                     padding=0,
                                     padding_mode='reflect')
        self.score = None        

    def forward(self, x):

        B, L, D = x.shape
        x_r = self.conv(x.permute(0, 2, 1)).transpose(1, 2)
        _, Lr, _ = x_r.shape
        #Q = torch.cat(self.W_q(x).chunk(self._h, dim=-1), dim=0)
        #K = torch.cat(self.W_k(x).chunk(self._h, dim=-1), dim=0)
        #V = torch.cat(self.W_v(x).chunk(self._h, dim=-1), dim=0)
        Q = self.W_q(x).reshape(B, L, self._h, self._q).permute(0, 2, 1, 3) #+ self.pe
        K = self.W_k(x_r).reshape(B, Lr, self._h, self._q).permute(0, 2, 1, 3) #+ self.pe
        V = self.W_v(x_r).reshape(B, Lr, self._h, self._q).permute(0, 2, 1, 3) #+ self.pe


        scores = torch.einsum('bhld, bhjd -> bhlj', Q, K)
        #scores_dh = torch.einsum('bhld, bhle -> bhde', Q, K)
        #scores_h = torch.einsum('bhld, bald -> bha', Q, K)
        #score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self._q)
        #score = torch.matmul(Q - K, (Q - K).transpose(-1, -2))
        # if self.mask:
        #     mask = torch.ones_like(scores[0]) #self.rel_pos.to(self.device) #torch.ones_like(scores[0])
        #     mask = torch.tril(mask, diagonal=0)
        #     scores = torch.where(mask > 0, scores, torch.Tensor([-2**32+1]).expand_as(scores[0]).to(self.device))

        #if self.mask:
        #    mask_dh = torch.ones_like(scores_dh[0]) #self.rel_pos.to(self.device) #torch.ones_like(scores[0])
        #    mask_dh = torch.tril(mask_dh, diagonal=0)
        #    scores_dh = torch.where(mask_dh > 0, scores_dh, torch.Tensor([-2**32+1]).expand_as(scores_dh[0]).to(self.device))
            

        scores = F.softmax(scores / math.sqrt(self._q), dim=-1) 
        self.score = scores
        #scores_dh = F.softmax(scores_dh / math.sqrt(L), dim=-1) 
        #scores_h = F.softmax(scores_h / math.sqrt(L*self._q), dim=-1) 
        #attention = torch.matmul(score, V) #.transpose(-1, -2))
        atten_out = torch.einsum('bhlj, bhjd -> bhld', scores, V)
        #atten_out_dh = torch.einsum('bhde, bhle -> bhld', scores_dh, V)
        #atten_out_h = torch.einsum('bha, bald -> bhld', scores_h, V)

        self_attention = self.W_o(atten_out.permute(0, 2, 3, 1).reshape(B, L, -1)) #+ self.pe
        #self_attention_dh = self.W_oh(atten_out_dh.permute(0, 2, 3, 1).reshape(B, L, -1)) #+ self.pe
        #self_attention_h = self.W_ohh(atten_out_h.permute(0, 2, 3, 1).reshape(B, L, -1))
        #attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        return self_attention, self.score   
   




class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

       
class Encoder(Module):
    def __init__(self,
                 d_model: int,
                 mha: Module,
                 feed_forward: Module,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 #patch_size: int,
                 device: str,
                 mask: bool = True,
                 dropout: float = 0):
        super(Encoder, self).__init__()

        self.MHA = mha #MultiHeadAttention(d_model=d_model, q=q, v=v, h=h, device=device, mask=mask, dropout=dropout)
        ###self.SMHA = Segment_Self_Attention_IntraSeg(d_model=d_model, q=q, v=v, h=h, mask=False, device=device, dropout=dropout)
        self.feedforward = FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layerNormal_1 = torch.nn.LayerNorm(d_model)
        self.layerNormal_2 = torch.nn.LayerNorm(d_model)
        #self.layerNormal_3 = layer_norm3   #torch.nn.LayerNorm(d_model)
        #self.a = torch.nn.Parameter(1e-5)

        self.device = device


        #for i in range(22):
        #    for j in range(22):
        #        self.rel_pos[0, 0, i, j] = torch.exp(-torch.pow(torch.Tensor([i - j]), 2) / 22).to(self.device)

    def forward(self, x):

        residual = x #+ self.rel_pos
        x, score = self.MHA(residual) #,,, score1, score2, score3, score4
        #x, score_s = self.MHA(residual)
        x = self.dropout(x)
        x = self.layerNormal_1(x + residual)
        ##residual = x
        ##x, score_s = self.SMHA(residual)
        ##x = self.layerNormal_3(x + residual)
        #x = x + residual

        residual = x
        x = self.feedforward(residual)
        x = self.dropout(x)
        x = self.layerNormal_2(x + residual)
        #x = x + residual

        return x, score


class FeedForward(Module):

    def __init__(self,
                 d_model: int,
                 d_hidden: int = 512):
        super(FeedForward, self).__init__()

        self.linear_1 = torch.nn.Linear(d_model, d_hidden)
        self.linear_2 = torch.nn.Linear(d_hidden, d_model)
        self.activation = torch.nn.GELU()  #torch.nn.SiLU()  #

    def forward(self, x):

        x = self.linear_1(x)
        x = self.activation(x)  #F.relu(x)
        x = self.linear_2(x)

        return x

