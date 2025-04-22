#!/usr/bin/env python3
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince
from env import R2RBatch
from model import EncoderLSTM, AttnDecoderLSTM
from agent import Seq2SeqAgent
from eval import Evaluation


TRAIN_VOCAB = 'seq2seq/data/train_vocab.txt'
TRAINVAL_VOCAB = 'seq2seq/data/trainval_vocab.txt'
RESULT_DIR = 'seq2seq/results/'
SNAPSHOT_DIR = 'seq2seq/snapshots/'
PLOT_DIR = 'seq2seq/plots/'
os.makedirs(os.path.dirname(RESULT_DIR), exist_ok=True)
os.makedirs(os.path.dirname(SNAPSHOT_DIR), exist_ok=True)
os.makedirs(os.path.dirname(PLOT_DIR), exist_ok=True)

IMAGENET_FEATURES = 'seq2seq/data/img_features/ResNet-152-imagenet.tsv'
MAX_INPUT_LENGTH = 80

features = IMAGENET_FEATURES
batch_size = 100
max_episode_len = 20
word_embedding_size = 256
action_embedding_size = 32
hidden_size = 512
bidirectional = False
dropout_ratio = 0.5
feedback_method = 'sample' # teacher or sample
learning_rate = 0.0001
weight_decay = 0.0005
n_iters = 5000 if feedback_method == 'teacher' else 20000
model_prefix = 'seq2seq_%s_imagenet' % (feedback_method)


def train(train_env, encoder, decoder, n_iters, log_every=100, val_envs={}):
    ''' Train on training set, validating on both seen and unseen. '''
    # 训练函数：在训练集上训练，同时在可见和不可见验证集上验证
    # train_env: 训练环境
    # encoder: 编码器
    # decoder: 解码器
    # n_iters: 训练迭代次数
    # log_every: 每log_every次迭代记录一次结果
    # val_envs: 验证环境，包含已见和未见两种场景

    #agent的封装了训练环境和模型
    agent = Seq2SeqAgent(train_env, "", encoder, decoder, max_episode_len)
    print('Training with %s feedback' % feedback_method)

    #两个模型各自的优化器
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 数据记录字典，用于跟踪训练过程中的各种指标
    # 数据结构：默认字典，键为指标名称，值为指标值列表
    data_log = defaultdict(list)
    start = time.time()

    for idx in range(0, n_iters, log_every):
        # 训练循环: 每log_every次迭代记录一次结果
        # 训练数据：指令序列(文本)和对应的动作序列

        interval = min(log_every,n_iters-idx)
        iter = idx + interval
        data_log['iteration'].append(iter)

        # 训练log_every次迭代
        # 输入: 文本指令序列和视觉特征
        # 输出: 预测的动作序列
        #单个 epoch的训练函数，包含了前向和反向传播
        agent.train(encoder_optimizer, decoder_optimizer, interval, feedback=feedback_method)
        #一些log
        train_losses = np.array(agent.losses)
        assert len(train_losses) == interval
        train_loss_avg = np.average(train_losses)
        data_log['train loss'].append(train_loss_avg)
        loss_str = 'train loss: %.4f' % train_loss_avg

        # 在验证集上运行评估
        # 每个epoch都会进行一次验证
        for env_name, (env, evaluator) in val_envs.items():
            agent.env = env
            agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, model_prefix, env_name, iter)
            
            # 有dropout的测试
            agent.test(use_dropout=True, feedback=feedback_method, allow_cheat=True)
            #一些log代码
            val_losses = np.array(agent.losses)
            val_loss_avg = np.average(val_losses)
            data_log['%s loss' % env_name].append(val_loss_avg)

            # 无dropout下的测试
            agent.test(use_dropout=False, feedback='argmax')
            agent.write_results() #把测试得到路径写入json文件 "results_path"
            #根据json文件中的路径计算评估指标
            score_summary, _ = evaluator.score(agent.results_path)
            loss_str += ', %s loss: %.4f' % (env_name, val_loss_avg)
            for metric,val in score_summary.items():
                data_log['%s %s' % (env_name,metric)].append(val)
                if metric in ['success_rate']:
                    loss_str += ', %s: %.3f' % (metric, val)

        agent.env = train_env

        print('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str))

        # 保存训练日志和模型快照
        # 关键指标：训练损失、验证损失、成功率等
        df = pd.DataFrame(data_log)
        df.set_index('iteration')
        df_path = '%s%s_log.csv' % (PLOT_DIR, model_prefix)
        df.to_csv(df_path)

        split_string = "-".join(train_env.splits)
        enc_path = '%s%s_%s_enc_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, iter)
        dec_path = '%s%s_%s_dec_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, iter)
        agent.save(enc_path, dec_path)


def setup():
    # 设置函数：初始化随机种子并检查词汇表
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train','val_seen','val_unseen']), TRAINVAL_VOCAB)


def test_submission():
    ''' Train on combined training and validation sets, and generate test submission. '''
    # 测试提交函数：在合并的训练和验证集上训练，并生成测试提交

    setup()
    # 创建批处理训练环境，同时预处理文本
    # 数据集结构：R2R数据集，包含指令和对应路径
    vocab = read_vocab(TRAINVAL_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    train_env = R2RBatch(features, batch_size=batch_size, splits=['train', 'val_seen', 'val_unseen'], tokenizer=tok)

    # 构建模型并训练
    # 模型：编码器-解码器架构，编码器处理文本指令，解码器生成导航动作
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx,
                  dropout_ratio, bidirectional=bidirectional).cuda()
    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                  action_embedding_size, hidden_size, dropout_ratio).cuda()
    train(train_env, encoder, decoder, n_iters)

    # 生成测试提交
    # 输出：代理在测试环境中执行的动作序列
    test_env = R2RBatch(features, batch_size=batch_size, splits=['test'], tokenizer=tok)
    agent = Seq2SeqAgent(test_env, "", encoder, decoder, max_episode_len)
    agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, model_prefix, 'test', 20000)
    agent.test(use_dropout=False, feedback='argmax')
    agent.write_results()


def train_val():
    ''' Train on the training set, and validate on seen and unseen splits. '''
    # 训练验证函数：在训练集上训练，在已见和未见的分割上验证

    setup()
    # 创建批处理训练环境，同时预处理文本
    # 词汇表：从训练数据构建
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    #创建训练环境
    train_env = R2RBatch(features, batch_size=batch_size, splits=['train'], tokenizer=tok)

    # 创建验证环境
    # 两种验证环境：已见场景(val_seen)和未见场景(val_unseen)
    val_envs = {split: (R2RBatch(features, batch_size=batch_size, splits=[split],
                tokenizer=tok), Evaluation([split])) for split in ['val_seen', 'val_unseen']}

    # 构建模型并训练
    # 核心逻辑：序列到序列模型，将自然语言指令转换为导航动作序列
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    # encoder的作用是什么？
    # encoder的作用是将输入的文本指令转换为高维的语义向量，为解码器提供文本信息
    # decoder的作用是什么？
    # decoder的作用是根据编码器提供的文本信息，生成导航动作序列
    encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx,
                  dropout_ratio, bidirectional=bidirectional).cuda()
    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                  action_embedding_size, hidden_size, dropout_ratio).cuda()
    # 训练主函数
    train(train_env, encoder, decoder, n_iters, val_envs=val_envs)


if __name__ == "__main__":
    train_val()
    #test_submission()
