import json
import os
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict
import line_profiler

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils.distributed import is_default_gpu
from utils.ops import pad_tensors, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence

from .agent_base import Seq2SeqAgent
from .eval_utils import cal_dtw

from models.graph_utils import GraphMap
from models.model import VLNBert, Critic
from models.ops import pad_tensors_wgrad


class GMapNavAgent(Seq2SeqAgent):
    
    def _build_model(self):
        #vln_bert 就是预训练阶段的模型
        self.vln_bert = VLNBert(self.args).cuda()
        self.critic = Critic(self.args).cuda()
        # buffer
        self.scanvp_cands = {}

    def _language_variable(self, obs):
        #仅仅是把指令的token转为tensor
        seq_lengths = [len(ob['instr_encoding']) for ob in obs]
        
        seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)
        mask = np.zeros((len(obs), max(seq_lengths)), dtype=np.bool)
        for i, ob in enumerate(obs):
            seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']
            mask[i, :seq_lengths[i]] = True

        seq_tensor = torch.from_numpy(seq_tensor).long().cuda()
        mask = torch.from_numpy(mask).cuda()
        return {
            'txt_ids': seq_tensor, 'txt_masks': mask
        }

    def _panorama_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        batch_view_img_fts, batch_loc_fts, batch_nav_types = [], [], []
        batch_view_lens, batch_cand_vpids = [], []
        
        for i, ob in enumerate(obs):
            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
            # cand views
            used_viewidxs = set()
            for j, cc in enumerate(ob['candidate']):
                #每个候选视点的特征从环境中获取，计算方式详见
                #duet/map_nav_src/r2r/env.py R2RNavBatch._get_obs
                #这里的特征是已经是经过视觉编码器提取的
                view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                nav_types.append(1)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])
            # non cand views
            view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
                in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
                in enumerate(ob['feature']) if k not in used_viewidxs])
            nav_types.extend([0] * (36 - len(used_viewidxs)))
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)
            
            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_loc_fts.append(torch.from_numpy(view_loc_fts))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))

        # pad features to max_len
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda() #(batch_size, 37, 768)
        #思考：这里为什是37？
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda() #(batch_size, 37, 7)
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'view_img_fts': batch_view_img_fts, 'loc_fts': batch_loc_fts, 
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens, 
            'cand_vpids': batch_cand_vpids,
        }

    def _nav_gmap_variable(self, obs, gmaps):
        # [stop] + gmap_vpids
        batch_size = len(obs)
        
        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []                
            for k in gmap.node_positions.keys():
                if self.args.act_visited_nodes:
                    if k == obs[i]['viewpoint']:
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
                else:
                    if gmap.graph.visited(k):
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
            batch_no_vp_left.append(len(unvisited_vpids) == 0)
            if self.args.enc_full_graph:
                gmap_vpids = [None] + visited_vpids + unvisited_vpids
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
            else:
                gmap_vpids = [None] + unvisited_vpids
                gmap_visited_masks = [0] * len(gmap_vpids)

            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]
            gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[1:]]
            gmap_img_embeds = torch.stack(
                [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
            )   # cuda

            gmap_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], gmap_vpids, obs[i]['heading'], obs[i]['elevation'],
            )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for i in range(1, len(gmap_vpids)):
                for j in range(i+1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.graph.distance(gmap_vpids[i], gmap_vpids[j])

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vpids': batch_gmap_vpids, 'gmap_img_embeds': batch_gmap_img_embeds, 
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_visited_masks': batch_gmap_visited_masks, 
            'gmap_pair_dists': gmap_pair_dists, 'gmap_masks': batch_gmap_masks,
            'no_vp_left': batch_no_vp_left,
        }

    def _nav_vp_variable(self, obs, gmaps, pano_embeds, cand_vpids, view_lens, nav_types):
        batch_size = len(obs)

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        batch_vp_pos_fts = []
        for i, gmap in enumerate(gmaps):
            cur_cand_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], cand_vpids[i], 
                obs[i]['heading'], obs[i]['elevation']
            )
            cur_start_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], [gmap.start_vp], 
                obs[i]['heading'], obs[i]['elevation']
            )                    
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts)+1, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))

        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()

        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), nav_types == 1], 1)

        return {
            'vp_img_embeds': vp_img_embeds,
            'vp_pos_fts': batch_vp_pos_fts,
            'vp_masks': gen_seq_masks(view_lens+1),
            'vp_nav_masks': vp_nav_masks,
            'vp_cand_vpids': [[None]+x for x in cand_vpids],
        }

    def _teacher_action(self, obs, vpids, ended, visited_masks=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if ob['viewpoint'] == ob['gt_path'][-1]:
                    a[i] = 0    # Stop if arrived 
                else:
                    scan = ob['scan']
                    cur_vp = ob['viewpoint']
                    min_idx, min_dist = self.args.ignoreid, float('inf')
                    for j, vpid in enumerate(vpids[i]):
                        if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                            # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                            dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                    + self.env.shortest_distances[scan][cur_vp][vpid]
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                    a[i] = min_idx
                    if min_idx == self.args.ignoreid:
                        print('scan %s: all vps are searched' % (scan))

        return torch.from_numpy(a).cuda()

    def _teacher_action_r4r(
        self, obs, vpids, ended, visited_masks=None, imitation_learning=False, t=None, traj=None
    ):
        """
        生成教师动作（专家轨迹）用于模仿学习
        
        R4R数据集不同于R2R，它使用的不一定是最短路径，目标位置可能是已访问过的节点。
        该函数根据不同模式（完全模仿学习或基于启发式的专家策略）生成最优动作。
        
        参数:
        - obs: 当前观察
        - vpids: 可选视点ID列表
        - ended: 是否已结束的标志
        - visited_masks: 已访问节点的掩码
        - imitation_learning: 是否使用严格的模仿学习模式
        - t: 当前时间步（用于模仿学习模式）
        - traj: 当前轨迹（用于启发式专家策略）
        
        返回:
        - 最优动作索引的张量
        """
        # 初始化动作数组，默认值为0
        a = np.zeros(len(obs), dtype=np.int64)
        
        # 针对批次中的每个样本计算教师动作
        for i, ob in enumerate(obs):
            # 如果样本已结束，则忽略（设置为ignoreid）
            if ended[i]:                                            # 如果导航已结束，忽略此样本
                a[i] = self.args.ignoreid
            else:
                # ===== 1. 模仿学习模式：严格按照标准路径操作 =====
                if imitation_learning:
                    # 确保当前位置与标准路径的当前步骤匹配
                    assert ob['viewpoint'] == ob['gt_path'][t]
                    
                    # 如果已到达标准路径的最后一步，选择停止动作（索引0）
                    if t == len(ob['gt_path']) - 1:
                        a[i] = 0    # 停止动作
                    else:
                        # 找到标准路径中的下一个视点
                        goal_vp = ob['gt_path'][t + 1]
                        
                        # 在当前可见视点中查找匹配的下一个视点
                        for j, vpid in enumerate(vpids[i]):
                            if goal_vp == vpid:
                                a[i] = j  # 找到匹配的视点索引
                                break
                
                # ===== 2. 启发式专家策略：根据距离指标选择最优动作 =====
                else:
                    # 如果已到达目标终点，选择停止动作
                    if ob['viewpoint'] == ob['gt_path'][-1]:
                        a[i] = 0    # 到达终点，选择停止
                    else:
                        # 获取当前场景ID和当前视点
                        scan = ob['scan']
                        cur_vp = ob['viewpoint']
                        
                        # 初始化用于寻找最优下一步的变量
                        min_idx, min_dist = self.args.ignoreid, float('inf')
                        
                        # 遍历所有可能的下一个视点, 计算每个视点到目标的距离，选择最小距离的视点作为最优下一步
                        for j, vpid in enumerate(vpids[i]):
                            # j>0 跳过停止动作；并且只考虑未访问的节点(或忽略访问状态)
                            if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                                
                                # ===== 2.1 基于NDTW的专家策略 =====
                                # NDTW (Normalized Dynamic Time Warping)衡量路径与标准路径的相似度
                                if self.args.expert_policy == 'ndtw':
                                    # 计算选择该视点后的完整路径
                                    predicted_path = sum(traj[i]['path'], []) + self.env.shortest_paths[scan][ob['viewpoint']][vpid][1:]
                                    
                                    # 计算与标准路径的NDTW距离（负值，因为越高越好）
                                    dist = - cal_dtw(
                                        self.env.shortest_distances[scan], 
                                        predicted_path, 
                                        ob['gt_path'], 
                                        threshold=3.0
                                    )['nDTW']
                                
                                # ===== 2.2 基于SPL的专家策略 =====
                                # SPL (Success weighted by Path Length)考虑路径长度最短
                                elif self.args.expert_policy == 'spl':
                                    # 计算从当前位置到目标终点的距离（经过候选视点）
                                    dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                            + self.env.shortest_distances[scan][cur_vp][vpid]
                                
                                # 选择距离最小的视点作为最优下一步
                                if dist < min_dist:
                                    min_dist = dist
                                    min_idx = j
                        
                        # 设置找到的最优视点索引
                        a[i] = min_idx
                        
                        # 如果没有找到合适的视点（所有视点都已访问）
                        if min_idx == self.args.ignoreid:
                            print('scan %s: all vps are searched' % (scan))
        
        # 将numpy数组转换为pytorch张量并移至GPU
        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, gmaps, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action is not None:            # None is the <stop> action
                traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], action))
                if len(traj[i]['path'][-1]) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                    prev_vp = traj[i]['path'][-1][-2]
                viewidx = self.scanvp_cands['%s_%s'%(ob['scan'], prev_vp)][action]
                heading = (viewidx % 12) * math.radians(30)
                elevation = (viewidx // 12 - 1) * math.radians(30)
                self.env.env.sims[i].newEpisode([ob['scan']], [action], [heading], [elevation])

    def _update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    # @profile
    def rollout(self, train_ml=None, train_rl=False, reset=True):
        """
        执行一次完整的导航过程
        
        参数:
        - train_ml: 模仿学习权重，用于监督学习
        - train_rl: 是否使用强化学习
        - reset: 是否重置环境
        
        返回:
        - traj: 导航轨迹记录
        """
        # ===== 1. 环境初始化阶段 =====
        if reset:  # 重置环境
            obs = self.env.reset()  # 获取环境初始状态和观察
        else:
            obs = self.env._get_obs()  # 不重置时，获取当前观察
        self._update_scanvp_cands(obs)  # 更新场景中视点的候选信息（记录可见视点）

        batch_size = len(obs)  # 批处理大小（同时处理的导航实例数量）
        
        # ===== 2. 构建认知地图 =====
        # 为每个样本创建一个导航图，初始包含起始视点
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]  # 创建表示环境的图结构
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)  # 根据初始观察更新图结构

        # ===== 3. 轨迹记录初始化 =====
        # 记录导航路径，包含指令ID和路径信息
        traj = [{
            'instr_id': ob['instr_id'],  # 指令ID，用于评估
            'path': [[ob['viewpoint']]],  # 初始路径只包含起始点
            'details': {},  # 用于记录详细信息
        } for ob in obs]

        # ===== 4. 语言指令处理 =====
        # 处理语言输入：将文本指令转换为模型可用的特征
        # 仅仅是把指令的token转为tensor
        language_inputs = self._language_variable(obs)  # 处理文本指令
        # 通过BERT提取文本特征
        txt_embeds = self.vln_bert('language', language_inputs)  # 通过BERT编码文本指令
    
        # ===== 5. 状态跟踪初始化 =====
        # 初始化状态跟踪变量
        ended = np.array([False] * batch_size)  # 标记每个样本是否结束导航
        just_ended = np.array([False] * batch_size)  # 标记刚刚结束的样本

        # 初始化日志记录
        masks = []
        entropys = []  # 用于记录动作选择的熵（表示不确定性）
        ml_loss = 0.  # 模仿学习损失初始化

        # ===== 6. 导航主循环 =====
        for t in range(self.args.max_action_len):  # 最大动作步数限制
            # 更新节点步骤ID（记录节点被访问的时间步）
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1  # 记录当前位置的访问时间步

            # ===== 6.1 全景特征处理 - 观察当前环境 =====
            # 提取全景特征表示 - 获取当前位置的视觉观察
            pano_inputs = self._panorama_feature_variable(obs)  # 提取全景图像特征
            #vln_bert 里包含了整个策略网络
            pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs)  # 编码全景视觉特征
            # 计算平均全景嵌入（整合当前视点的所有视角信息）
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            # ===== 6.2 更新环境认知图 =====
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # 更新当前已访问节点的嵌入表示
                    i_vp = obs[i]['viewpoint']
                    gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)  # 更新当前节点的表征
                    
                    # 更新未访问但可见的候选节点
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gmap.graph.visited(i_cand_vp):  # 如果节点未被访问过
                            gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])  # 更新其节点表示

            # ===== 6.3 多模态导航决策 =====
            # 构建导航决策所需的输入
            # gmap来构建全局和局部的表征
            #全局表征：visited viewpoint 和 candidate viewpoint，
            #如果一个candidate viewpoint在多个点都见过需要把这些节点的表征做个平均
            #局部表征：当前节点的视觉表征
            nav_inputs = self._nav_gmap_variable(obs, gmaps)  # 全局地图输入（宏观视角）
            nav_inputs.update(
                self._nav_vp_variable(  # 局部视点输入（微观视角）
                    obs, gmaps, pano_embeds, pano_inputs['cand_vpids'], 
                    pano_inputs['view_lens'], pano_inputs['nav_types'],
                )
            )
            # 添加语言指令信息，实现视觉-语言多模态融合
            nav_inputs.update({
                'txt_embeds': txt_embeds,  # 文本特征
                'txt_masks': language_inputs['txt_masks'],  # 文本掩码
            })
            
            # 通过VLN-BERT模型进行导航推理，生成下一步动作预测
            nav_outs = self.vln_bert('navigation', nav_inputs)

            # ===== 6.4 融合策略选择 =====
            # 根据不同的融合策略选择使用的导航逻辑和视点
            if self.args.fusion == 'local':  # 仅使用局部视点信息
                nav_logits = nav_outs['local_logits']  # 局部决策分数
                nav_vpids = nav_inputs['vp_cand_vpids']  # 局部候选视点
            elif self.args.fusion == 'global':  # 仅使用全局地图信息
                nav_logits = nav_outs['global_logits']  # 全局决策分数
                nav_vpids = nav_inputs['gmap_vpids']  # 全局候选视点
            else:  # 融合局部视点和全局地图信息（默认策略）
                nav_logits = nav_outs['fused_logits']  # 融合后的决策分数
                nav_vpids = nav_inputs['gmap_vpids']  # 使用全局视点列表

            # 将logits转换为概率分布
            nav_probs = torch.softmax(nav_logits, 1)  # 归一化得到动作概率
            
            # ===== 6.5 更新图中的停止得分 =====
            # 记录每个节点的停止概率（用于选择最佳停止位置）
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    i_vp = obs[i]['viewpoint']
                    gmap.node_stop_scores[i_vp] = {
                        'stop': nav_probs[i, 0].data.item(),  # 停止动作的概率
                    }
            
            # ===== 6.6 训练目标计算（模仿学习） =====                     
            if train_ml is not None:  # 如果是训练模式
                # 获取教师动作（监督学习的目标）
                if self.args.dataset == 'r2r':
                    # 调用_teacher_action_r4r获取最优动作
                    # 教师动作是指在当前状态下，专家会选择的最佳下一步动作
                    # 这些动作用作监督信号，指导模型学习如何导航
                    nav_targets = self._teacher_action_r4r(
                        obs,  # 当前观察
                        nav_vpids,  # 可用的视点列表
                        ended,  # 是否已结束的标志
                        # 已访问节点的掩码（全局模式下使用，避免重复访问）
                        visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                        # 是否使用严格的模仿学习，teacher模式下完全按照标准路径行进
                        imitation_learning=(self.feedback=='teacher'), 
                        t=t,  # 当前时间步
                        traj=traj  # 当前轨迹（用于计算NDTW评估指标）
                    )
                elif self.args.dataset == 'r4r':
                    # 同样的逻辑，适用于R4R数据集
                    nav_targets = self._teacher_action_r4r(
                        obs, nav_vpids, ended, 
                        visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                        imitation_learning=(self.feedback=='teacher'), t=t, traj=traj
                    )
                
                # 计算交叉熵损失：模型预测与教师动作之间的差距
                # nav_logits是模型预测的动作概率(未归一化)
                # nav_targets是教师认为的最优动作(真实标签)
                ml_loss += self.criterion(nav_logits, nav_targets)
                                                 
            # ===== 6.7 动作选择策略 =====
            # 根据不同的反馈模式确定下一步动作
            if self.feedback == 'teacher':  # 教师强制模式：直接使用标准答案
                # 教师强制(Teacher Forcing)：训练初期使用，完全按照专家路径行动
                # 这种模式下，智能体按照标准路径(ground truth path)严格执行
                # 优点：学习稳定，快速收敛；缺点：可能过拟合，缺乏探索
                a_t = nav_targets  # 使用教师动作（模仿学习）
            elif self.feedback == 'argmax':  # 贪心模式：选择最高概率的动作
                # 贪心模式：总是选择模型认为最可能的动作
                # 用于测试或训练后期，让模型自主做决策
                # 这种模式完全依赖模型的预测，没有任何探索性
                _, a_t = nav_logits.max(1)  # 选择概率最高的动作
                a_t = a_t.detach()  # 分离梯度
            elif self.feedback == 'sample':  # 采样模式：根据概率分布采样动作
                # 采样模式：根据模型预测的概率分布随机采样动作
                # 这种方式在训练中增加了随机性和探索性
                # 有助于避免局部最优，提高泛化能力
                c = torch.distributions.Categorical(nav_probs)  # 构建分类分布
                self.logs['entropy'].append(c.entropy().sum().item())  # 记录熵（用于日志）
                entropys.append(c.entropy())  # 记录熵（用于优化）
                a_t = c.sample().detach()  # 根据概率采样动作
            elif self.feedback == 'expl_sample':  # 探索性采样：随机探索与贪心结合
                # 探索性采样：贪心和随机探索的结合体
                # 大部分时间使用贪心策略，但有一定概率进行随机探索
                # 这是强化学习中探索与利用(Exploration vs. Exploitation)平衡的体现
                _, a_t = nav_probs.max(1)  # 默认选择最高概率动作
                # 随机决定是否探索（有一定概率进行随机探索）
                rand_explores = np.random.rand(batch_size, ) > self.args.expl_max_ratio
                if self.args.fusion == 'local':
                    cpu_nav_masks = nav_inputs['vp_nav_masks'].data.cpu().numpy()
                else:
                    cpu_nav_masks = (nav_inputs['gmap_masks'] * nav_inputs['gmap_visited_masks'].logical_not()).data.cpu().numpy()
                # 对需要探索的样本进行随机动作选择
                for i in range(batch_size):
                    if rand_explores[i]:  # 如果需要探索
                        cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]
                        a_t[i] = np.random.choice(cand_a_t)  # 随机选择一个可行动作
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')  # 不支持的反馈模式

            # ===== 6.8 停止动作判断 =====
            # 判断是否需要停止导航
            if self.feedback == 'teacher' or self.feedback == 'sample':  # 训练模式下的停止判断
                # 如果当前位置就是目标终点，则停止
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:  # 测试模式下的停止判断
                a_t_stop = a_t == 0  # 如果模型选择了停止动作（索引0）则停止

            # ===== 6.9 准备环境动作 =====
            # 将模型的决策转换为环境动作
            cpu_a_t = []  
            for i in range(batch_size):
                # 结束条件：停止动作、已结束、无可用视点或达到最大步数
                if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == self.args.max_action_len - 1):
                    cpu_a_t.append(None)  # None表示停止动作
                    just_ended[i] = True  # 标记为刚刚结束
                else:
                    cpu_a_t.append(nav_vpids[i][a_t[i]])  # 选择下一个视点ID

            # ===== 6.10 执行动作并更新环境 =====
            # 执行动作并更新轨迹记录
            self.make_equiv_action(cpu_a_t, gmaps, obs, traj)  # 将高级动作转换为模拟器动作并执行
            
            # ===== 6.11 寻找最佳停止节点 =====
            # 为刚刚结束的导航选择最佳停止节点（具有最高停止得分的节点）
            for i in range(batch_size):
                if (not ended[i]) and just_ended[i]:  # 对于刚刚结束的样本
                    # 寻找停止得分最高的节点
                    stop_node, stop_score = None, {'stop': -float('inf')}
                    for k, v in gmaps[i].node_stop_scores.items():
                        if v['stop'] > stop_score['stop']:
                            stop_score = v
                            stop_node = k
                    # 如果最佳停止节点不是当前节点，则添加到该节点的路径
                    if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                        traj[i]['path'].append(gmaps[i].graph.path(obs[i]['viewpoint'], stop_node))
                    # 记录详细输出信息（如果需要）
                    if self.args.detailed_output:
                        for k, v in gmaps[i].node_stop_scores.items():
                            traj[i]['details'][k] = {
                                'stop_prob': float(v['stop']),
                            }

            # ===== 6.12 获取新的观察并更新图 =====
            # 获取执行动作后的新观察
            obs = self.env._get_obs()  # 获取新的环境观察
            self._update_scanvp_cands(obs)  # 更新视点候选信息
            # 更新尚未结束的样本的图结构
            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)  # 基于新观察更新图结构

            # ===== 6.13 更新结束状态 =====
            # 更新样本的结束状态
            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            # ===== 6.14 提前退出检查 =====
            # 如果所有样本都结束，提前退出循环（优化计算效率）
            if ended.all():
                break

        # ===== 7. 计算最终损失（如果在训练模式） =====
        if train_ml is not None:
            ml_loss = ml_loss * train_ml / batch_size  # 根据训练权重缩放损失
            self.loss += ml_loss  # 累积到总损失
            self.logs['IL_loss'].append(ml_loss.item())  # 记录模仿学习损失

        # ===== 8. 返回完整导航轨迹 =====
        return traj  # 返回导航轨迹（用于评估和可视化）
