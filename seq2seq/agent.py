''' Agents: stop/random/shortest/seq2seq  '''

import json
import os
import sys
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.distributions as D
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from env import R2RBatch
from utils import padding_idx

class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

    def write_results(self):
        output = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def rollout(self):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self):
        self.env.reset_epoch()
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        #print('Testing %s' % self.__class__.__name__)
        looped = False
        while True:
            for traj in self.rollout():
                if traj['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[traj['instr_id']] = traj['path']
            if looped:
                break


class StopAgent(BaseAgent):
    ''' An agent that doesn't move! '''

    def rollout(self):
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in self.env.reset()]
        return traj


class RandomAgent(BaseAgent):
    ''' An agent that picks a random direction then tries to go straight for
        five viewpoint steps and then stops. '''

    def rollout(self):
        obs = self.env.reset()
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        self.steps = random.sample(range(-11,1), len(obs))
        ended = [False] * len(obs)
        for t in range(30):
            actions = []
            for i,ob in enumerate(obs):
                if self.steps[i] >= 5:
                    actions.append((0, 0, 0)) # do nothing, i.e. end
                    ended[i] = True
                elif self.steps[i] < 0:
                    actions.append((0, 1, 0)) # turn right (direction choosing)
                    self.steps[i] += 1
                elif len(ob['navigableLocations']) > 1:
                    actions.append((1, 0, 0)) # go forward
                    self.steps[i] += 1
                else:
                    actions.append((0, 1, 0)) # turn right until we can go forward
            obs = self.env.step(actions)
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
        return traj


class ShortestAgent(BaseAgent):
    ''' An agent that always takes the shortest path to goal. '''

    def rollout(self):
        obs = self.env.reset()
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        ended = np.array([False] * len(obs))
        while True:
            actions = [ob['teacher'] for ob in obs]
            obs = self.env.step(actions)
            for i,a in enumerate(actions):
                if a == (0, 0, 0):
                    ended[i] = True
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
            if ended.all():
                break
        return traj


class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''
    # 基于LSTM序列到序列模型（带注意力机制）的智能体

    # For now, the agent can't pick which forward move to make - just the one in the middle
    # 动作空间定义：模型可以执行的动作集合
    model_actions = ['left', 'right', 'up', 'down', 'forward', '<end>', '<start>', '<ignore>']
    # 环境动作：对应模型动作的具体运动参数(移动量, 转向量, 仰角量)
    env_actions = [
      (0,-1, 0), # left 左转
      (0, 1, 0), # right 右转
      (0, 0, 1), # up 抬头
      (0, 0,-1), # down 低头
      (1, 0, 0), # forward 前进
      (0, 0, 0), # <end> 结束
      (0, 0, 0), # <start> 开始
      (0, 0, 0)  # <ignore> 忽略
    ]
    # 训练反馈模式选项
    feedback_options = ['teacher', 'argmax', 'sample']

    def __init__(self, env, results_path, encoder, decoder, episode_len=20):
        # 初始化智能体：环境、结果路径、编码器、解码器和回合长度
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.encoder = encoder  # 文本编码器
        self.decoder = decoder  # 动作解码器
        self.episode_len = episode_len  # 单次导航最大步数
        self.losses = []  # 损失记录
        # 交叉熵损失函数，忽略<ignore>标记的位置
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.model_actions.index('<ignore>'))

    @staticmethod
    def n_inputs():
        # 解码器输入维度：动作种类数量
        return len(Seq2SeqAgent.model_actions)

    @staticmethod
    def n_outputs():
        # 解码器输出维度：动作种类数量减去<start>和<ignore>
        return len(Seq2SeqAgent.model_actions)-2 # Model doesn't output start or ignore

    def _sort_batch(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''
        # 从观察列表中提取指令并按长度降序排列（便于PyTorch打包处理）
        
        # 提取指令编码
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        # 计算序列长度（找到第一个padding_idx的位置）
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1] # 若无padding则为全长
        
        # 转换为PyTorch张量
        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # 按长度排序序列
        seq_lengths, perm_idx = seq_lengths.sort(0, True)
        sorted_tensor = seq_tensor[perm_idx]
        # 创建掩码标记padding位置
        mask = (sorted_tensor == padding_idx)[:,:seq_lengths[0]]

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.byte().cuda(), \
               list(seq_lengths), list(perm_idx)

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        # 提取预计算的特征到Variable中
        # 观察中包含场景的视觉特征
        feature_size = obs[0]['feature'].shape[0]
        features = np.empty((len(obs),feature_size), dtype=np.float32)
        for i,ob in enumerate(obs):
            features[i,:] = ob['feature']
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def _teacher_action(self, obs, ended):
        ''' Extract teacher actions into variable. '''
        # 提取教师动作（用于监督学习）
        a = torch.LongTensor(len(obs))
        for i,ob in enumerate(obs):
            # Supervised teacher only moves one axis at a time
            # 教师的动作一次只在一个轴上移动
            ix,heading_chg,elevation_chg = ob['teacher']
            if heading_chg > 0:
                a[i] = self.model_actions.index('right')
            elif heading_chg < 0:
                a[i] = self.model_actions.index('left')
            elif elevation_chg > 0:
                a[i] = self.model_actions.index('up')
            elif elevation_chg < 0:
                a[i] = self.model_actions.index('down')
            elif ix > 0:
                a[i] = self.model_actions.index('forward')
            elif ended[i]:
                a[i] = self.model_actions.index('<ignore>')
            else:
                a[i] = self.model_actions.index('<end>')
        return Variable(a, requires_grad=False).cuda()

    def rollout(self):
        # rollout方法：导航的前向传播，计算loss

        # 重置环境，获得初始obs
        obs = np.array(self.env.reset())
        # obs 数据结构 [{obs_1}, {obs_2}, {obs_3}, ...]
        # obs_1 数据结构 {'instr_id': '1', 
        # 'scan': 'mJXqzFtmKg4',
        # 'viewpoint': '37c2223d40cb4aedb1563e5e0c3a53e1', 
        #  'viewIndex': 13,
        # 'heading': 0.0, yaw angle
        # 'elevation': 0.0, 
        # 'feature': array([...], dtype=float32), shape=(2048,), 
        # 'navigableLocations': [mattersim.viewpoint object_1, mattersim.viewpoint object_2, ...], 
        # 'teacher': (0, 0, 0)},   groud truth action
        # 'instructions': 'Turn around and walk behind couch outside.',
        # 'instr_encoding': array([...], dtype=int32)}
        batch_size = len(obs)

        # 因为指令长度不一样，预处理成相同长度，并且打乱顺序
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        #seq.shape = (batch_size, max_seq_length), max_seq_length = 80
        #seq_mask.shape = (batch_size, max_seq_length), 0表示有效，1表示padding
        #seq_lengths.shape = (batch_size,), 每个元素表示每个序列的有效长度
        #perm_idx = [idx_1, idx_2, idx_3, ...], 每个元素表示每个序列在原始obs中的索引

        perm_obs = obs[perm_idx] # 根据perm_idx重排obs

        # 记录起点
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # 通过编码器前向传播，获得初始隐藏状态和记忆单元
        # ctx: 编码器输出的上下文向量，包含指令的语义信息
        # h_t, c_t: 解码器的初始隐藏状态和记忆单元
        ctx,h_t,c_t = self.encoder(seq, seq_lengths)
        # ctx.shape = (batch_size, max_seq_length, hidden_size), hidden_size = 512
        # h_t.shape = (batch_size, hidden_size), 
        # c_t.shape = (batch_size, hidden_size)

        # 初始动作为<start>
        a_t = Variable(torch.ones(batch_size).long() * self.model_actions.index('<start>'),
                    requires_grad=False).cuda()
        ended = np.array([False] * batch_size) # 标记每个导航是否结束

        # 执行序列展开并计算损失
        self.loss = 0
        env_action = [None] * batch_size
        for t in range(self.episode_len):
            # 获取当前观察的图像特征
            f_t = self._feature_variable(perm_obs) # Image features from obs
            # f_t.shape = (batch_size, feature_size), feature_size = 2048
            # 解码器单步前向传播
            # 输入：上一步动作a_t、当前观察特征f_t、上一步隐藏状态h_t和c_t、指令上下文ctx
            # 输出：更新的隐藏状态h_t和c_t、注意力权重alpha、动作概率分布logit
            h_t,c_t,alpha,logit = self.decoder(a_t.view(-1, 1), f_t, h_t, c_t, ctx, seq_mask)
            # logit.shape = (batch_size, 6), 6表示6个动作
            # h_t.shape = (batch_size, hidden_size), 
            # c_t.shape = (batch_size, hidden_size)

            # 屏蔽无法前进的位置（如墙壁）
            for i,ob in enumerate(perm_obs):
                if len(ob['navigableLocations']) <= 1:
                    logit[i, self.model_actions.index('forward')] = -float('inf')

            # 监督训练：计算与教师动作的损失
            # 把环境中的ground truth action转换为模型动作
            target = self._teacher_action(perm_obs, ended)
            self.loss += self.criterion(logit, target) #计算交叉墒损失，动作是6个离散的组合，可以看成是一个6分类问题

            # 确定下一步模型输入（根据不同的反馈策略）
            # 此时损失已经计算完成，这一步影响的是下一步模型会移动到什么地方
            # teacher action： 不管模型做出什么动作，下一步都按照ground truth 移动
            # argmax： 根据模型做出的动作移动，deterministic 策略
            # sample： 根据模型做出的动作移动，sample 策略
            if self.feedback == 'teacher':
                a_t = target                # teacher forcing
            elif self.feedback == 'argmax':
                _,a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
            elif self.feedback == 'sample':
                probs = F.softmax(logit, dim=1)
                m = D.Categorical(probs)
                a_t = m.sample()            # sampling an action from model
            else:
                sys.exit('Invalid feedback option')

            # 更新结束标记并执行环境动作
            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()
                #mask 已经结束的环境
                if action_idx == self.model_actions.index('<end>'):
                    ended[i] = True
                env_action[idx] = self.env_actions[action_idx]

            # 执行环境步骤并获取新观察
            obs = np.array(self.env.step(env_action))
            perm_obs = obs[perm_idx]

            # 保存轨迹输出
            for i,ob in enumerate(perm_obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))

            # 如果全部结束则提前退出
            if ended.all():
                break

        # 记录平均损失
        self.losses.append(self.loss.item() / self.episode_len)
        return traj

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False):
        ''' Evaluate once on each instruction in the current environment '''
        # 测试函数：在当前环境的每条指令上评估一次
        if not allow_cheat: # permitted for purpose of calculating validation loss only
            assert feedback in ['argmax', 'sample'] # 测试时不允许使用教师强制
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
        super(Seq2SeqAgent, self).test()

    def train(self, encoder_optimizer, decoder_optimizer, n_iters, feedback='teacher'):
        ''' Train for a given number of iterations '''
        # 训练函数：训练指定次数的迭代
        # 输入：编码器优化器、解码器优化器、迭代次数、反馈方式
        assert feedback in self.feedback_options
        self.feedback = feedback
        self.encoder.train()
        self.decoder.train()
        self.losses = []
        for iter in range(1, n_iters + 1):
            # 一次迭代的训练过程
            # 1. 清空梯度
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            # 2. 执行一次rollout（前向传播）
            self.rollout()
            # 3. 反向传播计算梯度
            self.loss.backward()
            # 4. 更新模型参数
            encoder_optimizer.step()
            decoder_optimizer.step()

    def save(self, encoder_path, decoder_path):
        ''' Snapshot models '''
        # 保存模型参数
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def load(self, encoder_path, decoder_path):
        ''' Loads parameters (but not training state) '''
        # 加载模型参数（不包括训练状态）
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
