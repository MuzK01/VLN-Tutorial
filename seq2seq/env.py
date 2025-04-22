''' Batched Room-to-Room navigation environment '''
# 批处理的房间到房间导航环境

import sys
sys.path.append('build')
import MatterSim
import csv
import numpy as np
import math
import base64
import json
import random
import networkx as nx

from utils import load_datasets, load_nav_graphs

csv.field_size_limit(sys.maxsize)

scan_dir = '/projects/VLN-Tutorial/duet/datasets/Matterport3D/v1_unzip_scans'
connectivity_dir = '/projects/VLN-Tutorial/Matterport3DSimulator/connectivity'

class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''
    # EnvBatch: MatterSim环境的批处理封装
    # 核心的属性：self.sim, self.features（每个节点的视觉feature）, 
    # 核心函数：
    #   # newEpisodes, 
    #   # getStates, 从sim中获得当前状态，从self.features中获得当前节点的特征
    #   # makeSimpleActions: 对外接口，接收动作的index
    #   # makeActions: 根据makeSimpleActions里转换的动作，执行sim的动作
    # Mattersim的动作空间是附近的可导航点，self.sim.getState()里包含了一个可导航点的列表
    # 这个列表中的可导航点排序规则与当前视野的夹角由小到大，
    # 当执行foward动作时，sim会导航到第列表中一个可导航点
    # 使用离散化视角和预训练特征

    def __init__(self, feature_store=None, enable_depth=False, batch_size=100):
        # 初始化批处理环境
        # feature_store: 图像特征存储路径
        # enable_depth: 是否启用深度信息
        # batch_size: 批处理大小
        if feature_store:
            print('Loading image features from %s' % feature_store)
            tsv_fieldnames = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']
            self.features = {}
            with open(feature_store, "rt") as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = tsv_fieldnames)
                for item in reader:
                    self.image_h = int(item['image_h'])
                    self.image_w = int(item['image_w'])
                    self.vfov = int(item['vfov'])
                    long_id = self._make_id(item['scanId'], item['viewpointId'])
                    # 加载预提取的图像特征
                    # 每个位置包含36个视角，每个特征维度为2048
                    self.features[long_id] = np.frombuffer(base64.b64decode(item['features']),
                            dtype=np.float32).reshape((36, 2048))
        else:
            print('Image features not provided')
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60

        enable_rendering = True if not feature_store else False
        enable_depth = enable_depth if not feature_store else False

        # 初始化MatterSim模拟器
        # MatterSim: Matterport3D环境的模拟器，用于导航
        self.batch_size = batch_size
        self.sim = MatterSim.Simulator()
        self.sim.setNavGraphPath(connectivity_dir)
        self.sim.setDatasetPath(scan_dir)
        self.sim.setDepthEnabled(enable_depth)
        self.sim.setRenderingEnabled(enable_rendering)
        self.sim.setDiscretizedViewingAngles(True)
        self.sim.setBatchSize(self.batch_size)
        self.sim.setCameraResolution(self.image_w, self.image_h)
        self.sim.setCameraVFOV(math.radians(self.vfov))
        self.sim.initialize()

    def _make_id(self, scanId, viewpointId):
        # 创建特征存储的唯一ID
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        # 创建新的导航回合
        # scanIds: 场景ID列表
        # viewpointIds: 视点ID列表
        # headings: 初始朝向列表
        self.sim.newEpisode(scanIds, viewpointIds, headings, [0]*self.batch_size)

    def getStates(self):
        ''' Get list of states augmented with precomputed image features. rgb field will be empty. '''
        # 获取增强了预计算图像特征的状态列表
        feature_states = []
        for state in self.sim.getState():
            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                # 获取当前视角的特征向量
                feature = self.features[long_id][state.viewIndex,:]
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
        return feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        # 执行动作：使用完整的状态依赖动作接口（批处理输入）
        # 每个动作元素都是一个(索引, 朝向, 仰角)元组
        ix = []
        heading = []
        elevation = []
        for i,h,e in actions:
            ix.append(int(i))
            heading.append(float(h))
            elevation.append(float(e))
        self.sim.makeAction(ix, heading, elevation)

    def makeSimpleActions(self, simple_indices):
        ''' Take an action using a simple interface: 0-forward, 1-turn left, 2-turn right, 3-look up, 4-look down.
            All viewpoint changes are 30 degrees. Forward, look up and look down may not succeed - check state.
            WARNING - Very likely this simple interface restricts some edges in the graph. Parts of the
            environment may not longer be navigable. '''
        # 使用简单接口执行动作：
        # 0-前进, 1-左转, 2-右转, 3-抬头, 4-低头
        actions = []
        for i, index in enumerate(simple_indices):
            if index == 0:
                actions.append((1, 0, 0))
            elif index == 1:
                actions.append((0,-1, 0))
            elif index == 2:
                actions.append((0, 1, 0))
            elif index == 3:
                actions.append((0, 0, 1))
            elif index == 4:
                actions.append((0, 0,-1))
            else:
                sys.exit("Invalid simple action");
        self.makeActions(actions)


class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''
    # R2RBatch: 实现房间到房间导航任务
    # 基本上是在EnvBatch的基础上按gym.env的接口封装了一层，增加了shorest_path_action 函数
    # 使用离散化视点和预训练特征

    def __init__(self, feature_store, enable_depth=False, batch_size=100, seed=10, splits=['train'], tokenizer=None):
        # 初始化R2R批处理环境
        # feature_store: 特征存储
        # splits: 数据集分割（训练、验证、测试）
        # tokenizer: 文本编码器
        self.feature_store = feature_store
        self.env = EnvBatch(feature_store=feature_store, enable_depth=enable_depth, batch_size=batch_size)
        self.data = []
        self.scans = []
        
        # 加载数据集
        # 数据集结构：
        # - scan: 场景ID
        # - path_id: 路径ID
        # - path: 路径中的视点序列
        # - instructions: 导航指令（自然语言）
        for item in load_datasets(splits):
            # 将多个指令拆分为单独的条目
            for j,instr in enumerate(item['instructions']):
                self.scans.append(item['scan'])
                new_item = dict(item)
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                new_item['instructions'] = instr
                if tokenizer:
                    # 对指令进行文本编码
                    new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
                self.data.append(new_item)
        self.scans = set(self.scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)
        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def _load_nav_graphs(self):
        ''' Load connectivity graph for each scan, useful for reasoning about shortest paths '''
        # 加载每个场景的连接图，用于计算最短路径
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            # 计算所有点对之间的最短路径
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            # 计算所有点对之间的最短距离
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self):
        # 获取下一个小批量数据
        batch = self.data[self.ix:self.ix+self.batch_size]
        if len(batch) < self.batch_size:
            random.shuffle(self.data)
            self.ix = self.batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += self.batch_size
        self.batch = batch

    def reset_epoch(self):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        # 将数据索引重置到时代开始，主要用于测试
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        # 确定通往目标的最短路径上的下一个动作，用于监督训练
        # 教师行为：为智能体提供基准动作
        if state.location.viewpointId == goalViewpointId:
            return (0, 0, 0) # 已到达目标，不做任何操作
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        # 能否看到下一个视点？
        for i,loc in enumerate(state.navigableLocations):
            if loc.viewpointId == nextViewpointId:
                # 移动前直接看向视点
                if loc.rel_heading > math.pi/6.0:
                      return (0, 1, 0) # 右转
                elif loc.rel_heading < -math.pi/6.0:
                      return (0,-1, 0) # 左转
                elif loc.rel_elevation > math.pi/6.0 and state.viewIndex//12 < 2:
                      return (0, 0, 1) # 抬头
                elif loc.rel_elevation < -math.pi/6.0 and state.viewIndex//12 > 0:
                      return (0, 0,-1) # 低头
                else:
                      return (i, 0, 0) # 移动
        # 看不见目标 - 首先调整相机仰角
        if state.viewIndex//12 == 0:
            return (0, 0, 1) # 抬头
        elif state.viewIndex//12 == 2:
            return (0, 0,-1) # 低头
        # 否则决定转向哪个方向
        pos = [state.location.x, state.location.y, state.location.z]
        target_rel = self.graphs[state.scanId].nodes[nextViewpointId]['position'] - pos
        target_heading = math.pi/2.0 - math.atan2(target_rel[1], target_rel[0]) # 转换为相对于y轴的角度
        if target_heading < 0:
            target_heading += 2.0*math.pi
        if state.heading > target_heading and state.heading - target_heading < math.pi:
            return (0,-1, 0) # 左转
        if target_heading > state.heading and target_heading - state.heading > math.pi:
            return (0,-1, 0) # 左转
        return (0, 1, 0) # 右转

    def _get_obs(self):
        # 获取观察
        # 输出：当前环境状态、导航指令和教师动作
        obs = []
        for i,(feature,state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            obs.append({
                'instr_id' : item['instr_id'],  # 指令ID
                'scan' : state.scanId,  # 场景ID
                'viewpoint' : state.location.viewpointId,  # 当前视点ID
                'viewIndex' : state.viewIndex,  # 视角索引
                'heading' : state.heading,  # 朝向
                'elevation' : state.elevation,  # 仰角
                'feature' : feature,  # 图像特征
                'step' : state.step,  # 当前步数
                'navigableLocations' : state.navigableLocations,  # 可导航位置
                'instructions' : item['instructions'],  # 自然语言指令
                'teacher' : self._shortest_path_action(state, item['path'][-1]),  # 教师动作
            })
            if not self.feature_store:
                obs[-1]['rgb'] = np.array(state.rgb)
                obs[-1]['depth'] = np.array(state.depth)
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']  # 指令编码
        return obs

    def reset(self):
        ''' Load a new minibatch / episodes. '''
        # 加载新的小批量/回合
        self._next_minibatch()
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]  # 路径起点
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        # 执行动作（与makeActions接口相同）
        # 输入：动作列表
        # 输出：执行动作后的观察
        self.env.makeActions(actions)
        return self._get_obs()
