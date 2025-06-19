import torch
import torch.nn as nn
from torch.nn.modules import ModuleList, Module
import math
from einops import rearrange, repeat


from model.MultiPatchFormer_Encoder import MultiHeadAttention, MultiHeadAttention_ch, Encoder, RevIN


class Model(nn.Module):
       
    def __init__(self, configs):

        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.d_channel = configs.enc_in
        # Embedding
        self.d_model = 256 
        self.d_hidden = 512
        self.q = 16
        self.v = 16
        self.h = 16
        self.device = 'cuda'
        self.mask = True
        self.dropout = 0.2
        self.stride1 = 8 
        self.patch_len1 = 8 
        self.stride2 = 8 
        self.patch_len2 = 16 
        self.stride3 = 7 
        self.patch_len3 = 24 
        self.stride4 = 6 
        self.patch_len4 = 32  
        self.patch_num1 = int((self.seq_len - self.patch_len2 )// self.stride2) + 2
        self.padding_patch_layer1 = nn.ReplicationPad1d((0, self.stride1))
        self.padding_patch_layer2 = nn.ReplicationPad1d((0, self.stride2))
        self.padding_patch_layer3 = nn.ReplicationPad1d((0, self.stride3))
        self.padding_patch_layer4 = nn.ReplicationPad1d((0, self.stride4))
        self.d_channel = configs.enc_in
        self.N = 1
       
        

        self.shared_MHA = nn.ModuleList([MultiHeadAttention(d_model=self.d_model, q=self.q, v=self.v, h=self.h, 
                                                           device=self.device, mask=self.mask, dropout=self.dropout,
                                                           ) for _ in range(self.N)])



        self.shared_MHA_ch = nn.ModuleList([MultiHeadAttention_ch(d_model=self.d_model, q=self.q, v=self.v, h=self.h,
                                                            device=self.device, mask=self.mask, dropout=self.dropout,
                                                         ) for _ in range(self.N)])
        

  
        
        self.shared_ff = 1 

        self.encoder_list = ModuleList([Encoder(d_model=self.d_model,
                                                  mha=self.shared_MHA[ll],
                                                  feed_forward=self.shared_ff,
                                                  d_hidden=self.d_hidden,
                                                  q=self.q,
                                                  v=self.v,
                                                  h=self.h,
                                                  #patch_size=2,
                                                  mask=True,
                                                  dropout=self.dropout,
                                                  device='cuda') for ll in range(self.N)])


        

        self.encoder_list_ch = ModuleList([Encoder(d_model=self.d_model,
                                                  mha=self.shared_MHA_ch[0],
                                                  feed_forward=self.shared_ff,
                                                  d_hidden=self.d_hidden,
                                                  q=self.q,
                                                  v=self.v,
                                                  h=self.h,
                                                  #patch_size=2,
                                                  mask=False,
                                                  dropout=self.dropout,
                                                  device='cuda') for ll in range(self.N)])



        
        
        pe = torch.zeros(self.patch_num1, self.d_model)
        for pos in range(self.patch_num1):
            for i in range(0, self.d_model, 2):
                wavelength = 10000 ** ((2 * i)/ self.d_model)
                pe[pos, i] = math.sin(pos / wavelength)
                pe[pos, i + 1] = math.cos(pos / wavelength)
        pe = pe.unsqueeze(0) # add a batch dimention to your pe matrix
        self.register_buffer('pe', pe)
       

        #self.embedding_patch_1 = torch.nn.Conv1d(in_channels=1, out_channels=self.d_model, kernel_size=self.patch_len1, stride=self.stride1)
        
        #self.embedding_patch_intra2i = torch.nn.Linear(self.d_channel, self.d_model)
        #self.embedding_patch_intra3i = torch.nn.Linear(self.d_channel, self.d_model)            
        #self.inter_patch_embed_2 = torch.nn.Linear(self.patch_num2, d_model)
        
        
        self.embedding_channel = nn.Conv1d(in_channels=self.d_model*self.patch_num1, out_channels=self.d_model, 
                                        kernel_size=1)#torch.nn.Linear(self.d_model*self.patch_num3, self.d_model)
        

        self.embedding_patch_1 = torch.nn.Conv1d(in_channels=1, out_channels=self.d_model//4, kernel_size=self.patch_len1, stride=self.stride1)
        self.embedding_patch_2 = torch.nn.Conv1d(in_channels=1, out_channels=self.d_model//4, kernel_size=self.patch_len2, stride=self.stride2)
        self.embedding_patch_3 = torch.nn.Conv1d(in_channels=1, out_channels=self.d_model//4, kernel_size=self.patch_len3, stride=self.stride3)
        self.embedding_patch_4 = torch.nn.Conv1d(in_channels=1, out_channels=self.d_model//4, kernel_size=self.patch_len4, stride=self.stride4)

        self.out_linear_1 = torch.nn.Linear(self.d_model, self.pred_len//8)
        self.out_linear_2 = torch.nn.Linear(self.d_model + self.pred_len//8, self.pred_len//8)
        self.out_linear_3 = torch.nn.Linear(self.d_model + 2*self.pred_len//8, self.pred_len//8)
        self.out_linear_4 = torch.nn.Linear(self.d_model + 3*self.pred_len//8, self.pred_len//8) #self.pred_len - 3*(self.pred_len//4))
        self.out_linear_5 = torch.nn.Linear(self.d_model + self.pred_len//2, self.pred_len//8)
        self.out_linear_6 = torch.nn.Linear(self.d_model + 5*self.pred_len//8, self.pred_len//8)
        self.out_linear_7 = torch.nn.Linear(self.d_model + 6*self.pred_len//8, self.pred_len//8)
        self.out_linear_8 = torch.nn.Linear(self.d_model + 7*self.pred_len//8, self.pred_len - 7*(self.pred_len//8))

        self.remap = torch.nn.Linear(self.d_model, self.seq_len)
        self.use_norm = configs.use_norm
        self.revin = RevIN(self.d_channel, affine=True)
        


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        if self.use_norm:
            x_enc_s = self.revin(x_enc, "norm")
            
        else:
            x_enc_s = x_enc
            
        x_i = x_enc_s.permute(0, 2, 1)
        
        x_i_p1 = x_i
        x_i_p2 = self.padding_patch_layer2(x_i)
        x_i_p3 = self.padding_patch_layer3(x_i)
        x_i_p4 = self.padding_patch_layer4(x_i)

        encoding_patch1 = self.embedding_patch_1(rearrange(x_i_p1, 'b c l -> (b c) l').unsqueeze(-1).permute(0, 2, 1)).permute(0, 2, 1) #+ self.pe
        encoding_patch2 = self.embedding_patch_2(rearrange(x_i_p2, 'b c l -> (b c) l').unsqueeze(-1).permute(0, 2, 1)).permute(0, 2, 1)
        encoding_patch3 = self.embedding_patch_3(rearrange(x_i_p3, 'b c l -> (b c) l').unsqueeze(-1).permute(0, 2, 1)).permute(0, 2, 1)
        encoding_patch4 = self.embedding_patch_4(rearrange(x_i_p4, 'b c l -> (b c) l').unsqueeze(-1).permute(0, 2, 1)).permute(0, 2, 1)
        
        encoding_patch = torch.cat((encoding_patch1, encoding_patch2, encoding_patch3, encoding_patch4
                                        ), dim=-1) + self.pe

        for i in range(self.N):

            encoding_patch = self.encoder_list[i](encoding_patch)[0] #+ encoding_patch# [(b c) p d]
            
  

        x_patch_c = rearrange(encoding_patch , '(b c) p d -> b c (p d)', b=x_enc.shape[0], c=self.d_channel)  # [(b c) p d] -> [(b c) p l]
        x_ch = self.embedding_channel(x_patch_c.permute(0, 2, 1)).transpose(1, 2)  # [b c d]
        
        encoding_1_ch = self.encoder_list_ch[0](x_ch)[0]  # [(b p) c d]
        
        #Semi Auto-regressive 
        forecast_ch1 = self.out_linear_1(encoding_1_ch)  #(batch, d_channel, d_out//4)
        forecast_ch2 = self.out_linear_2(torch.cat((encoding_1_ch, forecast_ch1), dim=-1))
        forecast_ch3 = self.out_linear_3(torch.cat((encoding_1_ch, forecast_ch1, forecast_ch2), dim=-1))
        forecast_ch4 = self.out_linear_4(torch.cat((encoding_1_ch, forecast_ch1, forecast_ch2, forecast_ch3), dim=-1))
        forecast_ch5 = self.out_linear_5(torch.cat((encoding_1_ch, forecast_ch1, forecast_ch2, 
                                                       forecast_ch3, forecast_ch4), dim=-1))
        forecast_ch6 = self.out_linear_6(torch.cat((encoding_1_ch, forecast_ch1, forecast_ch2, 
                                                       forecast_ch3, forecast_ch4, forecast_ch5), dim=-1))
        forecast_ch7 = self.out_linear_7(torch.cat((encoding_1_ch, forecast_ch1, forecast_ch2, 
                                                       forecast_ch3, forecast_ch4, forecast_ch5,
                                                       forecast_ch6), dim=-1))
        forecast_ch8 = self.out_linear_8(torch.cat((encoding_1_ch, forecast_ch1, forecast_ch2, 
                                                       forecast_ch3, forecast_ch4, forecast_ch5,
                                                       forecast_ch6, forecast_ch7), dim=-1))

        final_forecast = torch.cat((forecast_ch1, forecast_ch2, forecast_ch3, forecast_ch4, 
                                forecast_ch5, forecast_ch6, forecast_ch7, forecast_ch8
                                  ), dim=-1).permute(0, 2, 1)
    


        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            #dec_out_s = seasonal_forecast * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.d_output, 1))
            #dec_out_s = dec_out_s + (means[:, 0, :].unsqueeze(1).repeat(1, self.d_output, 1))
            dec_out_s = self.revin(final_forecast, "denorm")
            #dec_out_t = self.revin_t(trend_forecast, "denorm")
            return dec_out_s

        else:
            return tot_forecast
            #dec_out_t = trend_forecast * (stdev_t[:, 0, :].unsqueeze(1).repeat(1, self.d_output, 1))
            #dec_out_t = dec_out_t + (means_t[:, 0, :].unsqueeze(1).repeat(1, self.d_output, 1))
        



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
