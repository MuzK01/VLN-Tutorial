#!/usr/bin/env python3

''' Script to precompute image features using a Pytorch ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. '''

'''
此脚本用于预计算图像特征。它使用PyTorch中的神经网络（如ResNet或ViT），
在每个视点位置处获取36个离散视角的图像，每个视角间隔30度，
并使用指定的相机宽度、高度和垂直视场角参数。
'''

import os
import sys

import MatterSim

import argparse
import numpy as np
import json
import math
import h5py
import copy
from PIL import Image
import time
from progressbar import ProgressBar

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from utils import load_viewpoint_ids

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# 定义输出数据的字段名和特征维度
TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features', 'logits']
VIEWPOINT_SIZE = 36  # 每个视点的离散视角数量
FEATURE_SIZE = 768   # 特征向量的维度（对应ViT模型）
LOGIT_SIZE = 1000    # 分类输出的维度（ImageNet类别数）

# 相机参数设置
WIDTH = 640    # 图像宽度
HEIGHT = 480   # 图像高度
VFOV = 60      # 垂直视场角（度）


def build_feature_extractor(model_name, checkpoint_file=None):
    """
    构建特征提取器（神经网络模型）
    
    参数：
    - model_name: 模型名称（如'vit_base_patch16_224'）
    - checkpoint_file: 预训练模型文件路径（如果为None则使用默认预训练权重）
    
    返回：
    - model: 加载好的模型
    - img_transforms: 图像预处理函数
    - device: 运行设备（GPU或CPU）
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型并加载到设备
    model = timm.create_model(model_name, pretrained=(checkpoint_file is None)).to(device)
    if checkpoint_file is not None:
        state_dict = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)['state_dict']
        model.load_state_dict(state_dict)
    model.eval()  # 设置为评估模式

    # 创建图像预处理转换
    config = resolve_data_config({}, model=model)
    img_transforms = create_transform(**config)

    return model, img_transforms, device

def build_simulator(connectivity_dir, scan_dir):
    """
    构建Matterport3D环境模拟器
    
    参数：
    - connectivity_dir: 导航连接图目录
    - scan_dir: 3D扫描数据目录
    
    返回：
    - sim: 配置好的模拟器
    """
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim

def process_features(proc_id, out_queue, scanvp_list, args):
    """
    处理特定视点列表的特征提取（由多进程调用）
    
    参数：
    - proc_id: 进程ID
    - out_queue: 输出队列，用于存储处理结果
    - scanvp_list: 需处理的(scan_id, viewpoint_id)列表
    - args: 命令行参数
    """
    print('start proc_id: %d' % proc_id)

    # 设置模拟器
    sim = build_simulator(args.connectivity_dir, args.scan_dir)

    # 设置PyTorch CNN模型
    torch.set_grad_enabled(False)  # 禁用梯度计算，节省内存
    model, img_transforms, device = build_feature_extractor(args.model_name, args.checkpoint_file)

    # 遍历每个视点
    for scan_id, viewpoint_id in scanvp_list:
        # 采集该位置所有离散视角的图像
        images = []
        for ix in range(VIEWPOINT_SIZE):
            # 根据视角索引设置相机位置和朝向
            if ix == 0:
                # 初始视角：设置初始场景，朝向为0，垂直角度为-30度（向下看）
                sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(-30)])
            elif ix % 12 == 0:
                # 每12个视角（完成一周水平旋转后）：改变垂直视角
                sim.makeAction([0], [1.0], [1.0])
            else:
                # 其他情况：水平旋转30度
                sim.makeAction([0], [1.0], [0])
            
            # 获取当前状态和图像
            state = sim.getState()[0]
            assert state.viewIndex == ix

            # 处理图像（将BGR转换为RGB）
            image = np.array(state.rgb, copy=True)  # BGR格式
            image = Image.fromarray(image[:, :, ::-1])  # 转换为RGB
            images.append(image)

        # 对所有图像进行预处理并转为张量
        images = torch.stack([img_transforms(image).to(device) for image in images], 0)
        
        # 分批次提取特征，避免GPU内存溢出
        fts, logits = [], []
        for k in range(0, len(images), args.batch_size):
            # 提取特征向量
            b_fts = model.forward_features(images[k: k+args.batch_size])
            # 获取分类logits（用于图像分类的原始输出）
            b_logits = model.head(b_fts)
            
            # 转换为NumPy数组并保存
            b_fts = b_fts.data.cpu().numpy()
            b_logits = b_logits.data.cpu().numpy()
            fts.append(b_fts)
            logits.append(b_logits)
        
        # 合并所有批次的结果
        fts = np.concatenate(fts, 0)
        logits = np.concatenate(logits, 0)

        # 将结果放入输出队列
        out_queue.put((scan_id, viewpoint_id, fts, logits))

    # 标记此进程已完成
    out_queue.put(None)


def build_feature_file(args):
    """
    主处理函数：创建并存储所有视点的特征
    
    参数：
    - args: 命令行参数
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # 加载所有视点ID
    scanvp_list = load_viewpoint_ids(args.connectivity_dir)

    # 设置多进程
    num_workers = min(args.num_workers, len(scanvp_list))
    num_data_per_worker = len(scanvp_list) // num_workers

    # 创建输出队列和进程列表
    out_queue = mp.Queue()
    processes = []
    
    # 启动多个进程进行并行处理
    for proc_id in range(num_workers):
        # 为每个进程分配一部分数据
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = mp.Process(
            target=process_features,
            args=(proc_id, out_queue, scanvp_list[sidx: eidx], args)
        )
        process.start()
        processes.append(process)
    
    # 跟踪完成的进程和视点数量
    num_finished_workers = 0
    num_finished_vps = 0

    # 创建进度条
    progress_bar = ProgressBar(max_value=len(scanvp_list))
    progress_bar.start()

    # 将结果写入HDF5文件
    with h5py.File(args.output_file, 'w') as outf:
        while num_finished_workers < num_workers:
            # 从队列获取结果
            res = out_queue.get()
            if res is None:
                # 一个进程完成
                num_finished_workers += 1
            else:
                # 处理结果数据
                scan_id, viewpoint_id, fts, logits = res
                key = '%s_%s'%(scan_id, viewpoint_id)
                
                # 合并特征和logits（如果需要）
                if args.out_image_logits:
                    data = np.hstack([fts, logits])
                else:
                    data = fts
                
                # 创建数据集并写入数据
                outf.create_dataset(key, data.shape, dtype='float', compression='gzip')
                outf[key][...] = data
                
                # 添加元数据属性
                outf[key].attrs['scanId'] = scan_id
                outf[key].attrs['viewpointId'] = viewpoint_id
                outf[key].attrs['image_w'] = WIDTH
                outf[key].attrs['image_h'] = HEIGHT
                outf[key].attrs['vfov'] = VFOV

                # 更新进度
                num_finished_vps += 1
                progress_bar.update(num_finished_vps)

    # 完成进度条和清理进程
    progress_bar.finish()
    for process in processes:
        process.join()
            

if __name__ == '__main__':
    """
    主程序入口：解析命令行参数并开始处理
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='vit_base_patch16_224')  # 使用的模型名称
    parser.add_argument('--checkpoint_file', default=None)  # 预训练模型路径
    parser.add_argument('--connectivity_dir', default='../connectivity')  # 连接图目录
    parser.add_argument('--scan_dir', default='../data/v1/scans')  # 扫描数据目录
    parser.add_argument('--out_image_logits', action='store_true', default=False)  # 是否输出分类logits
    parser.add_argument('--output_file')  # 输出文件路径
    parser.add_argument('--batch_size', default=64, type=int)  # 批处理大小
    parser.add_argument('--num_workers', type=int, default=8)  # 进程数量
    args = parser.parse_args()

    # 开始处理
    build_feature_file(args)

#思考：对比直接存图片和直接存特征，两种方式的优劣和适用场景
#答：存特征计算效率高，但是无法训练视觉编码器
#存图片需要每一步再重新提取特征，效率低，需要对视觉编码器微调时使用

#练习
#尝试使用不同的特征编码器，对比他们的效果