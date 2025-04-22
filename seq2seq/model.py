import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''
    # EncoderLSTM：编码导航指令
    # 功能：将自然语言指令编码为上下文向量和解码器初始状态
    # 结构：词嵌入层 + LSTM + 线性映射层

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx, 
                            dropout_ratio, bidirectional=False, num_layers=1):
        # 初始化编码器
        # vocab_size: 词汇表大小
        # embedding_size: 词嵌入维度
        # hidden_size: LSTM隐藏状态维度
        # padding_idx: 填充标记的索引
        # dropout_ratio: Dropout比率
        # bidirectional: 是否使用双向LSTM
        # num_layers: LSTM层数
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)  # 词嵌入层
        self.lstm = nn.LSTM(embedding_size, hidden_size, self.num_layers, 
                            batch_first=True, dropout=dropout_ratio, 
                            bidirectional=bidirectional)  # LSTM层
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
            hidden_size * self.num_directions
        )  # 编码器到解码器的映射层

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        # 初始化LSTM的隐藏状态和记忆单元为零
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a 
            list of lengths for dynamic batching. '''
        # 前向传播：将输入的词索引序列编码为上下文向量和初始状态
        # inputs: 批次的词索引序列 (batch, seq_len)
        # lengths: 每个序列的实际长度，用于动态批处理
        # 输出:
        #   - ctx: 上下文向量，包含序列的所有隐藏状态 (batch, seq_len, hidden_size*num_directions)
        #   - decoder_init: 解码器的初始隐藏状态 (batch, hidden_size)
        #   - c_t: 解码器的初始记忆单元 (batch, hidden_size)
        
        # 1. 词嵌入
        embeds = self.embedding(inputs)   # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        
        # 2. 初始化LSTM状态
        h0, c0 = self.init_state(inputs)
        
        # 3. 打包序列以处理变长序列
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        
        # 4. LSTM前向传播
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        # 5. 处理双向LSTM的输出
        if self.num_directions == 2:
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)  # 连接双向LSTM的最后隐藏状态
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)  # 连接双向LSTM的最后记忆单元
        else:
            h_t = enc_h_t[-1]  # 单向LSTM的最后隐藏状态
            c_t = enc_c_t[-1]  # 单向LSTM的最后记忆单元

        # 6. 将编码器隐藏状态映射为解码器初始状态
        decoder_init = nn.Tanh()(self.encoder2decoder(h_t))

        # 7. 解包上下文向量
        ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
        ctx = self.drop(ctx)
        
        # 返回上下文向量、解码器初始隐藏状态和记忆单元
        return ctx, decoder_init, c_t


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention. 

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''
    # SoftDotAttention: 软点积注意力机制
    # 功能：计算查询向量与上下文向量的相似度，生成注意力权重
    # 结构：线性变换 + softmax + 加权组合

    def __init__(self, dim):
        '''Initialize layer.'''
        # 初始化注意力层
        # dim: 隐藏状态维度
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)  # 查询向量的线性变换
        self.sm = nn.Softmax(dim=1)  # softmax操作
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)  # 输出的线性变换
        self.tanh = nn.Tanh()  # tanh激活函数

    def forward(self, h, context, mask=None):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        # 前向传播：计算注意力权重和加权上下文向量
        # h: 当前解码器隐藏状态 (batch, dim)
        # context: 编码器的上下文向量 (batch, seq_len, dim)
        # mask: 掩码，指示哪些位置应被屏蔽 (batch, seq_len)
        # 输出:
        #   - h_tilde: 结合上下文的新隐藏状态 (batch, dim)
        #   - attn: 注意力权重 (batch, seq_len)
        
        # 1. 对查询向量进行线性变换并扩展维度
        target = self.linear_in(h).unsqueeze(2)  # (batch, dim, 1)

        # 2. 计算注意力分数：点积操作
        attn = torch.bmm(context, target).squeeze(2)  # (batch, seq_len)
        
        # 3. 应用掩码（如果提供）
        if mask is not None:
            # 在softmax之前，将掩码位置设为负无穷
            attn.data.masked_fill_(mask, -float('inf'))
        
        # 4. 应用softmax获取注意力权重
        attn = self.sm(attn)  # (batch, seq_len)
        
        # 5. 调整注意力权重的形状用于矩阵乘法
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # (batch, 1, seq_len)

        # 6. 使用注意力权重对上下文进行加权求和
        weighted_context = torch.bmm(attn3, context).squeeze(1)  # (batch, dim)
        
        # 7. 将加权上下文与当前隐藏状态连接
        h_tilde = torch.cat((weighted_context, h), 1)  # (batch, dim*2)

        # 8. 通过线性层和tanh激活函数生成新的隐藏状态
        h_tilde = self.tanh(self.linear_out(h_tilde))  # (batch, dim)
        
        return h_tilde, attn


class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''
    # AttnDecoderLSTM: 带注意力机制的LSTM解码器
    # 功能：解码导航动作，利用注意力机制关注指令的不同部分
    # 结构：动作嵌入 + LSTM + 注意力层 + 输出层

    def __init__(self, input_action_size, output_action_size, embedding_size, hidden_size, 
                      dropout_ratio, feature_size=2048):
        # 初始化解码器
        # input_action_size: 输入动作空间大小
        # output_action_size: 输出动作空间大小
        # embedding_size: 动作嵌入维度
        # hidden_size: LSTM隐藏状态维度
        # dropout_ratio: Dropout比率
        # feature_size: 视觉特征维度
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_action_size, embedding_size)  # 动作嵌入层
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)  # LSTM单元
        self.attention_layer = SoftDotAttention(hidden_size)  # 注意力层
        self.decoder2action = nn.Linear(hidden_size, output_action_size)  # 输出层

    def forward(self, action, feature, h_0, c_0, ctx, ctx_mask=None):
        ''' Takes a single step in the decoder LSTM (allowing sampling).

        action: batch x 1
        feature: batch x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''
        # 前向传播：解码器单步前向传播（支持采样）
        # action: 上一个动作索引 (batch, 1)
        # feature: 当前观察的视觉特征 (batch, feature_size)
        # h_0: 上一时刻的隐藏状态 (batch, hidden_size)
        # c_0: 上一时刻的记忆单元 (batch, hidden_size)
        # ctx: 编码器的上下文向量 (batch, seq_len, dim)
        # ctx_mask: 上下文掩码 (batch, seq_len)
        # 输出:
        #   - h_1: 更新的隐藏状态 (batch, hidden_size)
        #   - c_1: 更新的记忆单元 (batch, hidden_size)
        #   - alpha: 注意力权重 (batch, seq_len)
        #   - logit: 动作概率分布的对数 (batch, output_action_size)
        
        # 1. 对上一个动作进行嵌入
        action_embeds = self.embedding(action)   # (batch, 1, embedding_size)
        action_embeds = action_embeds.squeeze()  # (batch, embedding_size)
        
        # 2. 连接动作嵌入和视觉特征
        concat_input = torch.cat((action_embeds, feature), 1)  # (batch, embedding_size+feature_size)
        
        # 3. 应用dropout
        drop = self.drop(concat_input)
        
        # 4. LSTM单元更新
        h_1, c_1 = self.lstm(drop, (h_0, c_0))  # (batch, hidden_size), (batch, hidden_size)
        
        # 5. 对更新的隐藏状态应用dropout
        h_1_drop = self.drop(h_1)
        
        # 6. 应用注意力机制，获取新的隐藏状态和注意力权重
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)
        
        # 7. 将隐藏状态映射到动作空间
        logit = self.decoder2action(h_tilde)  # (batch, output_action_size)
        
        return h_1, c_1, alpha, logit


