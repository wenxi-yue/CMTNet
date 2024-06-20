import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math, copy
from torch.autograd import Variable



def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self,query, key, value , mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        
        nbatches = query.size(0)

        if len(value.size())>3:
            query = query.unsqueeze(2)
            query, key, value = \
                [l(x).view(nbatches, query.size(1), -1, self.h, self.d_k).permute(0,3,1,2,4)
                for l, x in zip(self.linears, (query, key, value))]
            
        
        else:
            query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)           

        if len(value.size())>4:
            x = x.permute(0,2,3,1,4).contiguous().view(nbatches, -1, 1, self.h * self.d_k).squeeze(-2)
        else:
            x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
    
        return self.linears[-1](x)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, ref, mask=None):
        for layer in self.layers:
            x = layer(x, ref, mask) 
        return self.norm(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.2):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
        
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout=0.2):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, ref, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, ref, ref, mask))
        
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=6000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):

        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)



class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



class conf_to_weight_channelonly(nn.Module):
    def __init__(self, num_classes):
        super(conf_to_weight_channelonly, self).__init__()

        self.max_pool = nn.MaxPool1d(num_classes)
        dimension = num_classes + 2

        self.linear_conf = nn.Linear(dimension, 1)
        
    def forward(self, conf):
        max_ = self.max_pool(conf)
        diff = (torch.topk(conf,k=2,dim=2)[0][:,:,0]-torch.topk(conf,k=2,dim=2)[0][:,:,1]).unsqueeze(-1)        
        conf = torch.cat((conf,max_,diff),dim=-1)

        return torch.sigmoid(self.linear_conf(conf))



class conf_to_weight(nn.Module):
    def __init__(self, num_classes):
        super(conf_to_weight, self).__init__()

        self.max_pool = nn.MaxPool1d(num_classes)
        dimension = num_classes + 2

        self.linear_conf = nn.Linear(dimension, 1)
        self.relu = nn.ReLU()
        
        self.dilation = 4
        
        self.kernal_1 = 3
        self.kernal_2 = 5 
        self.kernal_3 = 7

        self.conv_dilated_1 = nn.Conv1d(dimension,dimension,self.kernal_1, padding=(self.dilation*(self.kernal_1 - 1)),dilation=self.dilation)
        self.conv_dilated_2 = nn.Conv1d(dimension,dimension,self.kernal_2, padding=(self.dilation*(self.kernal_2 - 1)),dilation=self.dilation)
        self.conv_dilated_7 = nn.Conv1d(dimension,dimension,self.kernal_3, padding=(self.dilation*(self.kernal_3 - 1)),dilation=self.dilation)
        
        self.conv1 = nn.Conv1d(dimension, dimension, 1)
        self.conv2 = nn.Conv1d(dimension, dimension, 1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, conf):
        max_ = self.max_pool(conf)
        diff = (torch.topk(conf,k=2,dim=2)[0][:,:,0]-torch.topk(conf,k=2,dim=2)[0][:,:,1]).unsqueeze(-1)        
        conf = torch.cat((conf,max_,diff),dim=-1)
        
        conf = torch.transpose(conf,1,2)

        conf = self.relu(self.conv_dilated_1(conf))
        conf = conf[:, :, :-(self.dilation*(self.kernal_1 - 1))]

        conf = self.relu(self.conv_dilated_2(conf))
        conf = conf[:, :, :-(self.dilation*(self.kernal_2 - 1))]

        conf = self.relu(self.conv_dilated_7(conf))
        conf = conf[:, :, :-(self.dilation*(self.kernal_3 - 1))]
    
        conf = torch.transpose(conf,1,2)

        return torch.sigmoid(self.linear_conf(conf))

class Selective_Gated_Sum(nn.Module): 
    def __init__(self, d_model, num_classes):
        super(Selective_Gated_Sum, self).__init__()
        self.linear_decode = nn.Linear(d_model, num_classes)
        self.softmax = nn.Softmax(dim=-1)
        self.d_model = d_model
        
        self.conf_to_weight_model = conf_to_weight(num_classes)
        
        self.linear_weight = nn.Linear(3, 3)
        
        self.weight_map = None


    def forward(self, frame_context,phase_context,spatial_features):
        conf_spatial = self.softmax(self.linear_decode(spatial_features)) 
        conf_frame =self.softmax(self.linear_decode(frame_context)) 
        conf_phase = self.softmax(self.linear_decode(phase_context)) 
        
        weight_spatial = self.conf_to_weight_model(conf_spatial) 
        weight_frame = self.conf_to_weight_model(conf_frame)  
        weight_phase = self.conf_to_weight_model(conf_phase)   
        
        weight_map = torch.cat((weight_spatial.unsqueeze(2),weight_frame.unsqueeze(2),weight_phase.unsqueeze(2)),dim=-1)
        weight_map = self.linear_weight(weight_map)         
        weight_map = self.softmax(weight_map)      
        
        fused_feature =  weight_map*torch.cat((spatial_features.unsqueeze(-1),frame_context.unsqueeze(-1),phase_context.unsqueeze(-1)),dim=-1)            
        fused_feature = torch.sum(fused_feature,dim=-1)
        
        self.weight_map = weight_map
        
        return fused_feature
 
class Phase_Feature_Generation(torch.nn.Module):
    def __init__(self, d_model):
        super(Phase_Feature_Generation, self).__init__()
        self.norm = LayerNorm(d_model)
        
    def forward(self, out, ppm):
        T = out.size()[-2]
        ppm = torch.stack([ppm for _ in range(T)],dim=1)
        ppm = torch.mul(ppm,subsequent_mask(T).unsqueeze(3).cuda())
        ppm = torch.transpose(ppm,2,3)
        
        phase_features = torch.matmul(ppm,out.squeeze())
        
        phase_features = self.norm(phase_features)
        return phase_features
        

class AMCA(nn.Module):
    def __init__(self, encoder, phase_encoder, selective_gated_sum, phase_feature_generation):
        super(AMCA, self).__init__()
        self.encoder = encoder
        self.phaseencoder = phase_encoder
        self.selective_gated_sum = selective_gated_sum
        self.phase_feature_generation = phase_feature_generation
        self.up_sample = nn.Upsample(scale_factor=4, mode='nearest')

    def forward(self, spatial, out, src_mask, ppm):
      
        phase_features = self.phase_feature_generation(out[:,::4,:],ppm[:,::4,:])

        # generate frame-level context and phase-level context 
        frame_context = self.encoder(out, out, src_mask)
        
        phase_context = self.phaseencoder(out[:,::4,:],phase_features, None)
        phase_context = torch.transpose(self.up_sample(torch.transpose(phase_context,1,2)),1,2)
        phase_context = phase_context[:,:spatial.size()[1],:]

        # fuse frame-level context, phase-level context, and spatial feature using sgs
        fused_features = self.selective_gated_sum(frame_context,phase_context,spatial)
        return fused_features

class CMTNet(nn.Module):
    def __init__(self, N = 2 , d_model=128, d_ff=128 , num_classes = 7, M=2):
        super(CMTNet, self).__init__()
        single_stage = AMCA(Encoder(EncoderLayer(d_model, copy.deepcopy(MultiHeadedAttention(2, d_model)), copy.deepcopy(PositionwiseFeedForward(d_model, d_ff))), N),
                            Encoder(EncoderLayer(d_model, copy.deepcopy(MultiHeadedAttention(2, d_model)), copy.deepcopy(PositionwiseFeedForward(d_model, d_ff))), N),
                            Selective_Gated_Sum(d_model,num_classes),
                            Phase_Feature_Generation(d_model= d_model))
        
        self.stages = nn.ModuleList([copy.deepcopy(single_stage) for _ in range(M)])        
        self.linear = nn.Linear(2048, d_model)
        self.linear_cls = nn.Linear(d_model, num_classes)
        self.softmax = nn.Softmax(dim=-1)
        self.best_acc = -1
        self.best_epoch = 0
        self.val_loss_best_acc = 0 
        self.train_loss_best_acc = 0 
        self.train_acc_best_acc = 0 
        
    def forward(self, res_feature):
        src_mask = subsequent_mask(res_feature.size()[-2]).cuda()
        spatial_feature = self.linear(res_feature) 
        out_feature = spatial_feature
        logits = self.linear_cls(out_feature)
        ppm = self.softmax(logits)
        output = logits.unsqueeze(0)

        for each_stage in self.stages:
            out_feature = each_stage(spatial_feature,out_feature,src_mask,ppm)
            logits = self.linear_cls(out_feature)
            ppm = self.softmax(logits) 
            output = torch.cat((output,logits.unsqueeze(0)),dim=0)

        return output
