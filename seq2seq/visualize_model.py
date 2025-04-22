#!/usr/bin/env python3
# 用于创建模型可视化的脚本
# Script to create and save models for visualization with Netron

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os

# 导入模型定义
# Import model definitions
from model import EncoderLSTM, AttnDecoderLSTM, SoftDotAttention
SAVE_DIR = 'seq2seq/visualization'

def create_models_for_visualization():
    """创建模型并保存为可以在Netron中可视化的格式"""
    print("正在创建模型用于可视化...")
    print("Creating models for visualization...")
    
    # 创建保存目录
    # Create directory for saving models
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 设置设备为CPU
    # Set device to CPU for visualization
    device = torch.device("cpu")
    print(f"使用设备: {device}")
    print(f"Using device: {device}")
    
    # 设置模型参数
    # Set model parameters
    vocab_size = 1000  # 词汇表大小 / Vocabulary size
    embedding_size = 256  # 词嵌入维度 / Word embedding dimension
    hidden_size = 512  # 隐藏状态维度 / Hidden state dimension
    padding_idx = 0  # 填充标记索引 / Padding token index
    dropout_ratio = 0.0  # 设置为0避免警告 / Set to 0 to avoid warnings
    feature_size = 2048  # 视觉特征维度 / Visual feature dimension
    
    # 创建编码器
    # Create encoder
    encoder = EncoderLSTM(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        padding_idx=padding_idx,
        dropout_ratio=dropout_ratio,
        bidirectional=False,  # 单向LSTM / Unidirectional LSTM
        num_layers=1
    ).to(device)  # 将模型移到指定设备
    
    # 创建解码器
    # Create decoder
    decoder = AttnDecoderLSTM(
        input_action_size=8,  # 输入动作空间大小 / Input action space size
        output_action_size=6,  # 输出动作空间大小 / Output action space size
        embedding_size=32,  # 动作嵌入维度 / Action embedding dimension
        hidden_size=hidden_size,
        dropout_ratio=dropout_ratio,
        feature_size=feature_size
    ).to(device)  # 将模型移到指定设备
    
    # 创建示例输入
    # Create example inputs
    batch_size = 4
    seq_len = 20
    
    # 编码器输入：词索引和序列长度
    # Encoder inputs: word indices and sequence lengths
    encoder_inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    lengths = torch.tensor([seq_len] * batch_size, device=device)
    
    # 修改init_state方法内的device设置
    # Override the init_state method to use the same device
    original_init_state = encoder.init_state
    def new_init_state(inputs):
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            encoder.num_layers * encoder.num_directions,
            batch_size,
            encoder.hidden_size
        ), requires_grad=False).to(device)
        c0 = Variable(torch.zeros(
            encoder.num_layers * encoder.num_directions,
            batch_size,
            encoder.hidden_size
        ), requires_grad=False).to(device)
        return h0, c0
    
    # 暂时替换方法
    encoder.init_state = new_init_state
    
    # 使用torch.jit.trace跟踪和保存编码器
    # Trace and save encoder using torch.jit.trace
    try:
        # 必须将模型设置为评估模式以避免dropout
        # Must set model to eval mode to avoid dropout
        encoder.eval()
        
        # 跟踪编码器计算图
        # Trace encoder computation graph
        with torch.no_grad():
            traced_encoder = torch.jit.trace(
                encoder,
                (encoder_inputs, lengths)
            )
        
        # 保存跟踪后的模型
        # Save traced model
        traced_encoder.save(os.path.join(SAVE_DIR, 'encoder.pt'))
        print(f"编码器模型已保存为: {os.path.join(SAVE_DIR, 'encoder.pt')}")
        print(f"Encoder model saved as: {os.path.join(SAVE_DIR, 'encoder.pt')}")
        
        # 还可以使用torch.save保存模型
        # Can also save model using torch.save
        torch.save(encoder.state_dict(), os.path.join(SAVE_DIR, 'encoder_state_dict.pt'))
        torch.save(encoder, os.path.join(SAVE_DIR, 'encoder_full.pt'))
    except Exception as e:
        print(f"保存编码器时出错: {e}")
        print(f"Error saving encoder: {e}")
    finally:
        # 恢复原始方法
        encoder.init_state = original_init_state
    
    # 解码器输入
    # Decoder inputs
    action = torch.LongTensor([[0]] * batch_size).to(device)  # 批次大小为4的动作 / Actions for a batch of size 4
    feature = torch.randn(batch_size, feature_size, device=device)  # 随机视觉特征 / Random visual features
    h_0 = torch.randn(batch_size, hidden_size, device=device)  # 随机隐藏状态 / Random hidden state
    c_0 = torch.randn(batch_size, hidden_size, device=device)  # 随机记忆单元 / Random cell state
    ctx = torch.randn(batch_size, seq_len, hidden_size, device=device)  # 随机上下文向量 / Random context vector
    
    try:
        # 设置为评估模式
        # Set to eval mode
        decoder.eval()
        
        # 跟踪解码器计算图
        # Trace decoder computation graph
        with torch.no_grad():
            traced_decoder = torch.jit.trace(
                decoder,
                (action, feature, h_0, c_0, ctx)
            )
        
        # 保存跟踪后的模型
        # Save traced model
        traced_decoder.save(os.path.join(SAVE_DIR, 'decoder.pt'))
        print(f"解码器模型已保存为: {os.path.join(SAVE_DIR, 'decoder.pt')}")
        print(f"Decoder model saved as: {os.path.join(SAVE_DIR, 'decoder.pt')}")
        
        # 还可以使用torch.save保存模型
        # Can also save model using torch.save
        torch.save(decoder.state_dict(), os.path.join(SAVE_DIR, 'decoder_state_dict.pt'))
        torch.save(decoder, os.path.join(SAVE_DIR, 'decoder_full.pt'))
    except Exception as e:
        print(f"保存解码器时出错: {e}")
        print(f"Error saving decoder: {e}")
    
    print("\n要在Netron中可视化模型，请访问 https://netron.app 并上传保存的模型文件。")
    print("To visualize models in Netron, visit https://netron.app and upload the saved model files.")
    print("或者使用以下命令安装Netron: pip install netron")
    print("Or install Netron using: pip install netron")
    print("然后运行: netron " + os.path.join(SAVE_DIR, 'encoder.pt'))
    print("Then run: netron " + os.path.join(SAVE_DIR, 'encoder.pt'))

if __name__ == "__main__":
    create_models_for_visualization() 