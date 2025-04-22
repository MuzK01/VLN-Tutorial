''' Evaluation of agent trajectories '''

import json
import os
import sys
from collections import defaultdict
import networkx as nx
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

from env import R2RBatch
from utils import load_datasets, load_nav_graphs
from agent import BaseAgent, StopAgent, RandomAgent, ShortestAgent


class Evaluation(object):
    ''' 
    Results submission format:  [{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ] 
    # 评估类：用于评估导航智能体的轨迹表现
    # 输入：智能体生成的轨迹（viewpoint_id, heading, elevation）
    # 输出：导航错误、成功率、SPL等评估指标
    '''

    def __init__(self, splits):
        # 初始化评估器
        # 输入: splits - 数据集划分 (例如 train, val_seen, val_unseen)
        self.error_margin = 3.0  # 导航成功的误差容忍度 (3米)
        self.splits = splits
        self.gt = {}             # 存储真实路径数据
        self.instr_ids = []      # 所有指令ID
        self.scans = []          # 所有场景ID
        for item in load_datasets(splits):
            self.gt[item['path_id']] = item
            self.scans.append(item['scan'])
            self.instr_ids += ['%d_%d' % (item['path_id'],i) for i in range(3)]
        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)
        self.graphs = load_nav_graphs(self.scans)  # 加载导航图
        self.distances = {}
        for scan,G in self.graphs.items(): # 预计算所有最短路径距离
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
        #self.distances 以dict方式存储了所有scene的距离矩阵
        #单个scene的距离矩阵的结构如下：
        #{'80929af': {'baxsdfg': 12.3, '29dakzs': 12.3, ...}}
    def _get_nearest(self, scan, goal_id, path):
        # 找出路径中最接近目标的点
        # 输入: scan-场景ID, goal_id-目标点ID, path-智能体轨迹
        # 输出: 轨迹中离目标最近的点ID
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def _score_item(self, instr_id, path):
        # 评分单个轨迹
        # 输入: instr_id-指令ID, path-智能体轨迹
        # 处理: 计算导航错误、最优停止点错误、轨迹长度等
        ''' Calculate error based on the final position in trajectory, and also
            the closest position (oracle stopping rule). '''
        gt = self.gt[int(instr_id.split('_')[0])]
        #gt 的数据结构
        #{'path_id': 4370, 
        #   'scan': '2t7WbWqzZjQ', 
        #   'path': [''af3af33b0120469c9a00daa0d0b36799'', ...],
        #   'heading': 3.75
        #   'distance': 16.69
        #   'instructions': ['Go to the big red building.', ...]}
        start = gt['path'][0] # 起始点
        assert start == path[0][0], 'Result trajectories should include the start position'
        goal = gt['path'][-1] # 任务真实的目标点
        final_position = path[-1][0] # 智能体实际走到的目标点ID
        nearest_position = self._get_nearest(gt['scan'], goal, path) # 计算轨迹中最接近目标的节点ID
        # 计算最终位置到目标的距离误差
        self.scores['nav_errors'].append(self.distances[gt['scan']][final_position][goal])
        # 计算轨迹中最接近目标的位置误差(理想停止规则)
        self.scores['oracle_errors'].append(self.distances[gt['scan']][nearest_position][goal])
        distance = 0 # 计算完整路径的长度(米)
        prev = path[0]
        for curr in path[1:]:
            if prev[0] != curr[0]:
                try:
                    self.graphs[gt['scan']][prev[0]][curr[0]]
                except KeyError as err:
                    print('Error: The provided trajectory moves from %s to %s but the navigation graph contains no '\
                        'edge between these viewpoints. Please ensure the provided navigation trajectories '\
                        'are valid, so that trajectory length can be accurately calculated.' % (prev[0], curr[0]))
                    raise
            distance += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr
        self.scores['trajectory_lengths'].append(distance) # 智能体实际走的轨迹长度
        self.scores['shortest_path_lengths'].append(self.distances[gt['scan']][start][goal]) # 起点到终点的最短路径长度
        #self.scores 的结构
        #{'nav_errors': [12.3, 12.3, ...],
        # 'oracle_errors': [12.3, 12.3, ...],
        # 'trajectory_lengths': [12.3, 12.3, ...],
        # 'shortest_path_lengths': [12.3, 12.3, ...]}
        # 第n个列表元素对应第n个指令的评估结果

    def score(self, output_file):
        # 主评估函数
        # 输入: output_file-包含智能体轨迹的JSON文件
        # 输出: 评估指标汇总(成功率、平均导航错误、SPL等)
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids) # instr_ids 从数据集中加载， 是所有指令ID的集合
        with open(output_file) as f:
            for item in json.load(f):
                #item的数据结构案例
                #{instr_id: '4370_0', trajectory: [(viewpoint_id, heading_rads, elevation_rads),]}
                # 4370 是轨迹ID， 0是轨迹中的第1条指令，每条轨迹有3条指令
                # 检查结果中的指令ID是否在预期ID集合中
                if item['instr_id'] in instr_ids:
                    instr_ids.remove(item['instr_id']) # 从instr_ids中移除已处理的指令ID
                    self._score_item(item['instr_id'], item['trajectory']) # 以一个指令ID为最小单位进行评估， 结果存到self.scores中
        assert len(instr_ids) == 0, 'Trajectories not provided for %d instruction ids: %s' % (len(instr_ids),instr_ids)
        assert len(self.scores['nav_errors']) == len(self.instr_ids)
        num_successes = len([i for i in self.scores['nav_errors'] if i < self.error_margin])
        oracle_successes = len([i for i in self.scores['oracle_errors'] if i < self.error_margin])

        # 计算SPL (Success weighted by Path Length) - 路径长度加权的成功率
        spls = []
        for err,length,sp in zip(self.scores['nav_errors'],self.scores['trajectory_lengths'],self.scores['shortest_path_lengths']):
            if err < self.error_margin:
                spls.append(sp/max(length,sp)) #成功时，spl=最短路径/实际走的路径 （对比SR=1）
            else:
                spls.append(0) #失败时，spl=0

        # 评估指标汇总
        score_summary ={
            'length': np.average(self.scores['trajectory_lengths']),        # 平均轨迹长度
            'nav_error': np.average(self.scores['nav_errors']),             # 平均导航错误
            'oracle success_rate': float(oracle_successes)/float(len(self.scores['oracle_errors'])),  # 理想停止规则下的成功率
            'success_rate': float(num_successes)/float(len(self.scores['nav_errors'])),               # 实际成功率
            'spl': np.average(spls)                                         # 平均SPL指标
        }

        assert score_summary['spl'] <= score_summary['success_rate']
        return score_summary, self.scores


RESULT_DIR = 'seq2seq/results/'
os.makedirs(os.path.dirname(RESULT_DIR), exist_ok=True)

def eval_simple_agents():
    ''' Run simple baselines on each split. '''
    for split in ['train', 'val_seen', 'val_unseen']:
        env = R2RBatch(None, batch_size=1, splits=[split])
        ev = Evaluation([split])

        for agent_type in ['Stop', 'Shortest', 'Random']:
            outfile = '%s%s_%s_agent.json' % (RESULT_DIR, split, agent_type.lower())
            # agent = BaseAgent.get_agent(agent_type)(env, outfile) # 把env作为参数传入
            # agent.test() # 测试函数入口
            # agent.write_results() # 写入结果到RESULT_DIR
            score_summary, _ = ev.score(outfile) # 有了outfile后，计算评估指标
            print('\n%s' % agent_type)
            pp.pprint(score_summary)


def eval_seq2seq():
    ''' Eval sequence to sequence models on val splits (iteration selected from training error) '''
    outfiles = [
        RESULT_DIR + 'seq2seq_teacher_imagenet_%s_iter_5000.json',
        RESULT_DIR + 'seq2seq_sample_imagenet_%s_iter_20000.json'
    ]
    for outfile in outfiles:
        for split in ['val_seen', 'val_unseen']:
            ev = Evaluation([split])
            score_summary, _ = ev.score(outfile % split)
            print('\n%s' % outfile)
            pp.pprint(score_summary)


if __name__ == '__main__':

    eval_simple_agents()
    eval_seq2seq()
    # 对于训练的模型，每个epoch之后评测一次。





